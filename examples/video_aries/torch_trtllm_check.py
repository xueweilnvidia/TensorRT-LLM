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
                 config,
                 debug_mode=True,
                 stream: torch.cuda.Stream = None):
        self.dtype = config['pretrained_config']['dtype']

        rank = tensorrt_llm.mpi_rank()
        world_size = config['pretrained_config']['mapping']['world_size']
        cp_size = config['pretrained_config']['mapping']['cp_size']
        tp_size = config['pretrained_config']['mapping']['tp_size']
        pp_size = config['pretrained_config']['mapping']['pp_size']
        assert pp_size == 1
        self.mapping = tensorrt_llm.Mapping(world_size=world_size,
                                            rank=rank,
                                            cp_size=cp_size,
                                            tp_size=tp_size,
                                            pp_size=1,
                                            gpus_per_node=args.gpus_per_node)

        self.attn_num_heads = config['pretrained_config']['num_attention_heads']//tp_size
        local_rank = rank % self.mapping.gpus_per_node
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)
        CUASSERT(cudart.cudaSetDevice(local_rank))

        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        engine_file = os.path.join(args.tllm_model_dir, f"rank{rank}.engine")
        logger.info(f'Loading engine from {engine_file}')
        with open(engine_file, "rb") as f:
            engine_buffer = f.read()
        self.session = Session.from_serialized_engine(engine_buffer)
        print("Trtllm session created")

        self.debug_mode = debug_mode

        self.inputs = {}
        self.outputs = {}
        self.buffer_allocated = False

        expected_tensor_names = ['x', 't', 'text_states', 'text_mask', 'text_states_2', 'freqs_cos', 'freqs_sin', 'attn_mask', 'guidance', 'noise_pred']

        found_tensor_names = [
            self.session.engine.get_tensor_name(i)
            for i in range(self.session.engine.num_io_tensors)
        ]
        if self.debug_mode:
            self.debug_tensors = list(
                    set(found_tensor_names) - set(expected_tensor_names))
        if False:
            if rank==0:
                print(f"trt engine found io tensors:{found_tensor_names}")
                print(f"debug tensors:{self.debug_tensors}")
    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.session.engine.get_tensor_dtype(name))
        #if "input_len" in name:
        #    dtype = torch.int32
        return dtype

    def _io_buffer_setup(self, output_shape):
        model_inputs = ['x', 't', 'text_states', 'text_mask', 'text_states_2', 'freqs_cos', 'freqs_sin', 'attn_mask', 'guidance']

        for i in range(self.session.engine.num_io_tensors):
            name = self.session.engine.get_tensor_name(i)
            if self.debug_mode:
                mode = self.session.engine.get_tensor_mode(name)
                shape = self.session.engine.get_tensor_shape(name)
                dtype = self.session.engine.get_tensor_dtype(name)
                print(f"trt engine Tensor:name={name}, {mode}, {shape}, {dtype}")

            if name not in model_inputs:
                engine_output_shape = list(self.session.engine.get_tensor_shape(name))
                if name == "noise_pred":
                    self.outputs[name] = torch.empty(output_shape,
                                                 dtype=self._tensor_dtype(name),
                                                 device=self.device)

                if self.debug_mode and name in self.debug_tensors:
                    if 'img_before_double_block_debug_output' in name:
                        engine_output_shape=[1, 118800, 3072]
                    if 'bert_atn_debug_output' in name:
                        engine_output_shape=[1, 119056, 3072//8]
                    if 'qkv_debug_output' in name:
                        engine_output_shape=[1, 119056, 3072//8*3]
                    if 'single_block_input_x' in name:
                        engine_output_shape=[1, 119056, 3072]
                    if 'final_layer' in name:
                        engine_output_shape=[1, 118800, 64]
                    self.outputs[name] = torch.empty(engine_output_shape,
                                                 dtype=self._tensor_dtype(name),
                                                 device=self.device)
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
        attn_mask,torch.Size([2, 1, 8572, 8572]), torch.bool 
        freqs_cos,torch.Size([8316, 128]), torch.float32, 
        freqs_sin,torch.Size([8316, 128]), torch.float32, 
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
        return self.outputs['noise_pred']

def compute_mask(prompt_mask, img_seq_len):
   batch_size = prompt_mask.shape[0]
   txt_seq_len = prompt_mask.shape[1]
   seq_len = txt_seq_len+img_seq_len

   img_mask = torch.ones(batch_size, img_seq_len, dtype=torch.int64, device="cpu")
   # batch_size x seq_len
   concat_mask = torch.cat([img_mask, prompt_mask], dim=1)
   #attn_mask_1 = concat_mask.view(batch_size, 1, 1, seq_len).expand(batch_size, 1, seq_len, seq_len)
   attn_mask_1 = concat_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
   # batch_size x 1 x seq_len x seq_len
   attn_mask_2 = attn_mask_1.transpose(2, 3)
   # batch_size x 1 x seq_len x seq_len, 1 for broadcasting of num_heads
   attn_mask = (attn_mask_1 & attn_mask_2).bool()
   # avoids self-attention weight being NaN for text padding tokens
   attn_mask[:, :, :, 0]= True
   attn_mask = torch.where(attn_mask == 0, float('-inf'), 0.0).to(torch.bfloat16)
   attn_mask = attn_mask.expand(batch_size, 24, seq_len, seq_len)
   return attn_mask

def main(args):
    tensorrt_llm.logger.set_level(args.log_level)

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    assert torch.cuda.is_available()
    rank = tensorrt_llm.mpi_rank()

    file_path=args.input_data
    loaded_tensors = torch.load(file_path)

    if rank == 0:
        print("===========================io_tensors=========================")
        for key, tensor in loaded_tensors.items():
            print(f"{key},{tensor.shape}, {tensor.dtype} {tensor.device}")

        print("===============================================================")
    #import pdb;pdb.set_trace()
    # Load model:
    config_file = os.path.join(args.tllm_model_dir, 'config.json')
    with open(config_file) as f:
        config = json.load(f)
    model = TllmFlux(config, debug_mode=args.debug_mode)

    local_rank = rank % model.mapping.gpus_per_node
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    if args.attn_mode == "ootb" and ("attn_mask" not in loaded_tensors):
        img_seq_len = loaded_tensors["freqs_cos"].shape[0]
        attn_mask = compute_mask(loaded_tensors["text_mask"], img_seq_len)
        loaded_tensors["attn_mask"] = attn_mask
    if args.attn_mode == "ootb" and ("attn_mask" in loaded_tensors):
        if loaded_tensors["attn_mask"].dtype != torch.bool:
            attn_mask = loaded_tensors["attn_mask"]
            if model.mapping.cp_size > 1:
                attn_mask_cp_chunks = torch.chunk(attn_mask, model.mapping.cp_size, dim=2)
                attn_mask = attn_mask_cp_chunks[rank]
            if model.mapping.tp_size > 1:
                attn_mask_tp_chunks = torch.chunk(attn_mask, model.mapping.tp_size,  dim=1)
                attn_mask = attn_mask_tp_chunks[rank]
            loaded_tensors["attn_mask"]= attn_mask.contiguous()

    if args.attn_mode == "plugin" and ("attn_mask" in loaded_tensors):
        loaded_tensors.pop("attn_mask")

    model_inputs = dict()
    for key, tensor in loaded_tensors.items():
        if key not in ['output', 'noise_pred']:
            if rank == 0 and key == 'attn_mask':
                print(f"==========attn_mask,{tensor.shape}, {tensor.dtype}, {tensor.device}")
            model_inputs[key] = tensor.to(device)

    if rank == 0:
        print("===========================real model inputs=========================")
        for key, tensor in model_inputs.items():
            print(f"{key},{tensor.shape}, {tensor.dtype} {tensor.device}")

    output = model.forward(**model_inputs)
    if rank == 0:
        print(f"trtllm output shape:{output.shape}")

    non_mask = torch.isnan(output)
    num_nan = torch.sum(non_mask).item()
    if rank==0:
        print("trtllm_noise_pred tensor 中NaN值的数量:", num_nan)

    torch_noise_pred = loaded_tensors["noise_pred"].to(device)
    if rank==0:
        print(f"torch output shape:{torch_noise_pred.shape}")
    torch_output = torch_noise_pred.reshape(1,-1)
    trtllm_output = output.reshape(1,-1)
    diff= torch_output - trtllm_output
    if rank==0:
        print("diff:", diff)
        print("output:",'cosine_similarity:', torch.cosine_similarity(torch_output, trtllm_output).item(), 'mean diff:', diff.abs().mean().item(), 'max diff:', diff.abs().max())

        if model.debug_mode:
            torch.cuda.synchronize()
            '''
            print('-------input--------')
            for k, v in self.inputs.items():
                print(k, v.shape, v.dtype, v.device)
            '''
            print('-------output--------')
            for k, v in model.outputs.items():
                print(k, v.shape, v.dtype, "nan_num:",torch.isnan(v).sum().item(), "v.sum()=", v.sum())
                if 'double_blocks.0.core_attention.DoubleStreamBlock0_qkv_debug_output' in k:
                    torch.save(v, "trtllm_qkv_double_block_0.pt")
                if 'double_blocks.0.core_attention.DoubleStreamBlock0_bert_atn_debug_output' in k:
                    torch.save(v, "trtllm_atn_output_double_block_0.pt")
                if 'double_blocks.0.core_attention.DoubleStreamBlock0_input_lengths_debug_output' in k:
                    torch.save(v, "trtllm_inputlength_double_block_0.pt")


#os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tllm_model_dir",
                        type=str,
                        default='./engine_outputs/')
    parser.add_argument("--input_data",
                        type=str,
                        default='/jizhicfs/flyerxu/batch1_720p_io_tensors_step0.pt')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument("--debug_mode", type=bool, default=False)
    parser.add_argument('--attn_mode', type=str, default='plugin')
    args = parser.parse_args()
    main(args)
