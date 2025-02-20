"""
Sample new images from a pre-trained DiT.
"""
import json
import os

import tensorrt as trt
import torch
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
from functools import wraps

from cuda import cudart
from torchvision.utils import save_image

import tensorrt_llm
#from tensorrt_llm._ipc_utils import set_peer_access
from tensorrt_llm._utils import str_dtype_to_torch, trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper
from tensorrt_llm.runtime.session import Session, TensorInfo


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None


class TllmFlux(object):

    def __init__(self,
                 trtllm_model_dir,
                 debug_mode=False,
                 gpus_per_node=8,
                 stream: torch.cuda.Stream = None):
        config_file = os.path.join(trtllm_model_dir, 'config.json')
        with open(config_file) as f:
            config = json.load(f)
        self.config = config

        self.dtype = config['pretrained_config']['dtype']

        rank = tensorrt_llm.mpi_rank()
        world_size = config['pretrained_config']['mapping']['world_size']
        cp_size = config['pretrained_config']['mapping']['cp_size']
        tp_size = config['pretrained_config']['mapping']['tp_size']
        pp_size = config['pretrained_config']['mapping']['pp_size']

        self.attn_num_heads = config['pretrained_config']['num_attention_heads']//tp_size
        assert pp_size == 1
        self.mapping = tensorrt_llm.Mapping(world_size=world_size,
                                            rank=rank,
                                            cp_size=cp_size,
                                            tp_size=tp_size,
                                            pp_size=1,
                                            gpus_per_node=gpus_per_node)

        local_rank = rank % self.mapping.gpus_per_node
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)
        CUASSERT(cudart.cudaSetDevice(local_rank))

        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        engine_file = os.path.join(trtllm_model_dir, f"rank{rank}.engine")
        logger.info(f'Loading engine from {engine_file}')
        with open(engine_file, "rb") as f:
            engine_buffer = f.read()
        self.session = Session.from_serialized_engine(engine_buffer)
        logger.info("Trtllm session created")

        self.debug_mode = debug_mode

        self.inputs = {}
        self.outputs = {}
        self.buffer_allocated = False

        #expected_tensor_names = ['x', 't', 'text_sstates', 'text_mask', 'text_states_2', 'freqs_cos', 'freqs_sin', 'attn_mask', 'noise_pred']

        if self.debug_mode:
            found_tensor_names = [
                    self.session.engine.get_tensor_name(i)
                    for i in range(self.session.engine.num_io_tensors)
                    ]
            logger.info(f"engine found tensor names:{found_tensor_names}")
    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.session.engine.get_tensor_dtype(name))
        return dtype

    def _io_buffer_setup(self, output_shape):
        model_inputs = ['x', 't', 'text_states', 'text_mask', 'text_states_2', 'freqs_cos', 'freqs_sin', 'attn_mask', 'guidance']

        for i in range(self.session.engine.num_io_tensors):
            name = self.session.engine.get_tensor_name(i)
            if name not in model_inputs:
                #shape = list(self.session.engine.get_tensor_shape(name))
                self.outputs[name] = torch.empty(output_shape,
                                                 dtype=self._tensor_dtype(name),
                                                 device=self.device)

            if self.debug_mode:
                mode = self.session.engine.get_tensor_mode(name)
                shape = self.session.engine.get_tensor_shape(name)
                dtype = self.session.engine.get_tensor_dtype(name)
                print(f"trt engine Tenosr:name={name}, {mode}, {shape}, {dtype}")
        #self.outputs['output_attn_input_len']=torch.empty([output_shape[0]], dtype=torch.int32, device=self.device)
        self.buffer_allocated = True

    def cuda_stream_guard(func):
        """Sync external stream and set current stream to the one bound to the session. Reset on exit.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            external_stream = torch.cuda.current_stream()
            if external_stream != self.stream:
                external_stream.synchronize()
                torch.cuda.set_stream(self.stream)
            ret = func(self, *args, **kwargs)
            if external_stream != self.stream:
                self.stream.synchronize()
                torch.cuda.set_stream(external_stream)
            return ret

        return wrapper

    @cuda_stream_guard
    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                text_states: torch.Tensor,
                text_mask: torch.Tensor,
                text_states_2: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
                attn_mask: torch.Tensor = None,
                guidance: torch.Tensor = None,
                ):
        """
        x,torch.Size([2, 16, 33, 24, 42]), torch.float16 
        t,torch.Size([2]), torch.float32
        text_states,torch.Size([2, 256, 4096]), torch.float16 
        text_mask,torch.Size([2, 256]), torch.int64
        text_states_2,torch.Size([2, 768]), torch.float16 
        freqs_cos,torch.Size([8316, 128]), torch.float32, 
        freqs_sin,torch.Size([8316, 128]), torch.float32, 
        attn_mask,torch.Size([2, 1, 8572, 8572]), torch.bool 
        noise_pred,torch.Size([2, 16, 33, 24, 42]), torch.bfloat16,
        """

        self._io_buffer_setup(x.shape)
        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            attn_mask = torch.where(attn_mask == 0, float('-inf'), 0.0).to(str_dtype_to_torch(self.dtype))
            attn_mask = attn_mask.expand(attn_mask.shape[0], self.attn_num_heads, attn_mask.shape[2], attn_mask.shape[3])
        inputs = {
            "x": x.contiguous().to(str_dtype_to_torch(self.dtype)),
            "t": t.contiguous().to(str_dtype_to_torch("float32")),
            "text_states": text_states.contiguous().to(str_dtype_to_torch(self.dtype)),
            "text_mask": text_mask.int().contiguous(),
            "text_states_2": text_states_2.contiguous().to(str_dtype_to_torch(self.dtype)),
            "freqs_cos": freqs_cos.contiguous().to(str_dtype_to_torch("float32")),
            "freqs_sin": freqs_sin.contiguous().to(str_dtype_to_torch("float32")),
        }
        if attn_mask is not None:
            #inputs["attn_mask"]= attn_mask.contiguous().to(str_dtype_to_torch(self.dtype))
            assert attn_mask.dtype == str_dtype_to_torch(self.dtype)
            inputs["attn_mask"]= attn_mask.contiguous()
        if guidance is not None:
            inputs["guidance"] = guidance.contiguous().to(str_dtype_to_torch(self.dtype))
        self.inputs.update(**inputs)
        self.session.set_shapes(self.inputs)
        ok = self.session.run(self.inputs, self.outputs,
                              self.stream.cuda_stream)
        if not ok:
            raise RuntimeError('Executing TRTLLM engine failed!')
        if self.debug_mode:
            torch.cuda.synchronize()
            print('-------input--------')
            for k, v in self.inputs.items():
                print(k, v.shape, v.dtype, v.device)
            print('-------output--------')
            for k, v in self.outputs.items():
                print(k, v.shape, v.dtype, v.sum())
        return self.outputs['noise_pred']

