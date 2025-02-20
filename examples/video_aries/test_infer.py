"""
Sample new images from a pre-trained DiT.
"""
import json
import os

import torch
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
from functools import wraps

from cuda import cudart
from torchvision.utils import save_image

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, trt_dtype_to_torch
from tensorrt_llm.logger import logger
from Trtllm_flux_model import TllmFlux


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
    local_rank = rank % args.gpus_per_node
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    model = TllmFlux(args.tllm_model_dir, debug_mode=args.debug_mode)

    if args.attn_mode == "ootb" and ("attn_mask" not in loaded_tensors):
        img_seq_len = loaded_tensors["freqs_cos"].shape[0]
        attn_mask = compute_mask(loaded_tensors[text_mask], img_seq_len)
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
            model_inputs[key] = tensor.to(device)


    if rank == 0:
        print("===========================real model inputs=========================")
        for key, tensor in model_inputs.items():
            print(f"{key},{tensor.shape}, {tensor.dtype} {tensor.device}")

    output = model.forward(**model_inputs)
    if rank == 0:
        print(f"trtllm output shape:{output.shape}")


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()  # 开始记录时间
    run_num=args.run_num
    from tqdm import tqdm
    for _ in tqdm(range(run_num)):
        output = model.forward(**model_inputs)
    end_event.record()  # 结束记录时间
    torch.cuda.synchronize()  # 确保所有GPU操作完成
    total_time_ms = start_event.elapsed_time(end_event)
    elapsed_time_ms = total_time_ms/run_num  # 获取时间差，单位为毫秒
    if rank == 0:
        print(f"TRTLLM model infer average took: {elapsed_time_ms} milliseconds, {run_num} total time:{total_time_ms} ms")

#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tllm_model_dir",
                        type=str,
                        default='./engine_outputs/')
    parser.add_argument("--input_data",
                        type=str,
                        default='/jizhicfs/flyerxu/batch1_720p_io_tensors_step0.pt')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_num", type=int, default=35)
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument("--debug_mode", type=bool, default=False)
    parser.add_argument('--attn_mode', type=str, default='plugin')
    args = parser.parse_args()
    main(args)
