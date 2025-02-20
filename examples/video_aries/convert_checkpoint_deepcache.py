import argparse
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import safetensors.torch
import torch

import tensorrt_llm
from tensorrt_llm import str_dtype_to_torch
import tensorrt_llm.profiler
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import (split, split_matrix_tp,
                                               split_qkv_bias_tp, split_qkv_tp)


NAME_MAPPING = {
    'mlp.0' : 'mlp1',
    'mlp.2' : 'mlp2',
    'mlp.fc1' : 'mlp.fc',
    'mlp.fc2' : 'mlp.proj',
    'adaLN_modulation.1' : 'adaLN_modulation',
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flux_ckpt',
                        type=str,
                        default="/jizhicfs/flyerxu/video_flux.pt")
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument('--input_h',
                        type=int,
                        default=192,
                        help='The video height')
    parser.add_argument('--input_w',
                        type=int,
                        default=336,
                        help='The video width')
    parser.add_argument('--frames',
                        type=int,
                        default=129,
                        help='The video frames')
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--cp_size',
                        type=int,
                        default=1,
                        help='Context parallelism size')
    parser.add_argument('--cp_use_pad',
                        action='store_true',
                        help='when input hw cant not divie 8,cp_use_pad set to true')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument('--dtype',
                        type=str,
                        default='bfloat16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--fp8_linear',
                        action='store_true',
                        help='Whether use FP8 for linear layers')
    parser.add_argument('--fp8_scale_ckpt',
                        type=str,
                        default="/jizhicfs/flyerxu/fp8/fp8_video_flux_weight_scale.pt")
    parser.add_argument('--num_heads',
                        type=int,
                        default=24,
                        help='The number of heads of attention module')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=3072,
                        help='The hidden size of Flux')
    parser.add_argument('--attn_mode',
                        type=str,
                        default="plugin",
                        choices=['plugin', 'ootb'])
    parser.add_argument('--guidance_embed',
                        action='store_true',
                        help='for distilled model, use guidance_embed')

    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    parser.add_argument('--log_level',
                        type=str,
                        default='info',
                        choices=['internal_error', 'error', 'warning', 'info', 'verbose', 'debug'])

    return parser.parse_args()

def convert_video_flux(args, mapping:Mapping, config_dir, dtype='bfloat16'):
    weights = dict()
    torch_dtype = str_dtype_to_torch(dtype)
    tensor_parallel = mapping.tp_size
    tik = time.time()
    tensorrt_llm.profiler.print_memory_usage(f'Before weights loaded.', 'MiB')
    model_params = dict(torch.load(args.flux_ckpt, map_location=torch.device('cpu')))
    if 'module' in model_params:
        model_params = model_params['module']
    if args.fp8_scale_ckpt and args.fp8_linear:
        fp8_scale_params = dict(torch.load(args.fp8_scale_ckpt, map_location=torch.device('cpu')))
        model_params.update(fp8_scale_params)
    torch_to_trtllm_name = NAME_MAPPING

    with open(os.path.join(config_dir, 'config.json'), 'r') as f:
        model_config = json.loads(f.read())

    hidden_size = model_config['hidden_size']
    num_heads = model_config['num_heads']
    mlp_hidden_dim = int(model_config['hidden_size'] * model_config['mlp_ratio'])
    print(f"hidden_size={hidden_size},num_heads={num_heads}, mlp_hidden_dim={mlp_hidden_dim}")

    def get_trtllm_name(timm_name:str):
        for k, v in torch_to_trtllm_name.items():
            if k in timm_name:
                return 'model.' + timm_name.replace(k, v)
        return 'model.' + timm_name

    for name, param in model_params.items():
        trt_name = get_trtllm_name(name)
        if param.dtype == torch.int8 or 'scaling_factor' in name:
            if 'scaling_factor' in name:
                assert param.dtype == torch.float32
            weights[trt_name] = param.contiguous()
            print(f"load scaling_factor {trt_name}, {param}")
        else:
            weights[trt_name] = param.contiguous().to(torch_dtype)

    assert len(weights) == len(model_params)
    #to do tp/cp split
    if tensor_parallel > 1:
        print(f"=============weight tp slice, tp={tensor_parallel}")
        for k, v in weights.items():
            if re.match('^model.txt_in.individual_token_refiner.blocks.*.self_attn_qkv.weight$', k):
                weights[k] = split_qkv_tp(v, args.num_heads, args.hidden_size,
                                          tensor_parallel, mapping.tp_rank)
            elif re.match('^model.txt_in.individual_token_refiner.blocks.*.self_attn_qkv.bias$', k):
                weights[k] = split_qkv_bias_tp(v, args.num_heads, args.hidden_size,
                                               tensor_parallel, mapping.tp_rank)
            elif re.match('^model.txt_in.individual_token_refiner.blocks.*.self_attn_proj.weight$', k):
                weights[k] = split_matrix_tp(v,
                                             tensor_parallel,
                                             mapping.tp_rank,
                                             dim=1)
            elif re.match('^model.txt_in.individual_token_refiner.blocks.*.mlp.fc.weight$', k):
                weights[k] = split_matrix_tp(v,
                                             tensor_parallel,
                                             mapping.tp_rank,
                                             dim=0)
            elif re.match('^model.txt_in.individual_token_refiner.blocks.*.mlp.fc.bias$', k):
                weights[k] = split(v, tensor_parallel, mapping.tp_rank)
            elif re.match('^model.txt_in.individual_token_refiner.blocks.*.mlp.proj.weight$', k):
                weights[k] = split_matrix_tp(v,
                                             tensor_parallel,
                                             mapping.tp_rank,
                                             dim=1)
            elif re.match(r'.*adaLN_modulation.weight$', k):
                weights[k] = split_matrix_tp(v,
                                             tensor_parallel,
                                             mapping.tp_rank,
                                             dim=0)
            elif re.match(r'.*adaLN_modulation.bias$', k):
                weights[k] = split(v, tensor_parallel, mapping.tp_rank)

            elif re.match(r'^model.double_blocks.*.img_mod.linear.weight$', k):
                weights[k] = split_matrix_tp(v,
                                             tensor_parallel,
                                             mapping.tp_rank,
                                             dim=0)
            elif re.match(r'^model.double_blocks.*.img_mod.linear.bias$', k):
                weights[k] = split(v, tensor_parallel, mapping.tp_rank)

            elif re.match(r'^model.double_blocks.*.txt_mod.linear.weight$', k):
                weights[k] = split_matrix_tp(v,
                                             tensor_parallel,
                                             mapping.tp_rank,
                                             dim=0)
            elif re.match(r'^model.double_blocks.*.txt_mod.linear.bias$', k):
                weights[k] = split(v, tensor_parallel, mapping.tp_rank)

            elif re.match(r'^model.double_blocks.*.*_attn_qkv.weight$', k):
                weights[k] = split_qkv_tp(v, args.num_heads, args.hidden_size,
                                          tensor_parallel, mapping.tp_rank)
            elif re.match(r'^model.double_blocks.*.*_attn_qkv.bias$', k):
                weights[k] = split_qkv_bias_tp(v, args.num_heads, args.hidden_size,
                                               tensor_parallel, mapping.tp_rank)

            elif re.match('^model.double_blocks.*.*_attn_proj.weight$', k):
                weights[k] = split_matrix_tp(v,
                                             tensor_parallel,
                                             mapping.tp_rank,
                                             dim=1)
            elif re.match('^model.double_blocks.*.*_mlp.fc.weight$', k):
                weights[k] = split_matrix_tp(v,
                                             tensor_parallel,
                                             mapping.tp_rank,
                                             dim=0)
            elif re.match('^model.double_blocks.*.*_mlp.fc.bias$', k):
                weights[k] = split(v, tensor_parallel, mapping.tp_rank)
            elif re.match('^model.double_blocks.*.*_mlp.proj.weight$', k):
                weights[k] = split_matrix_tp(v,
                                             tensor_parallel,
                                             mapping.tp_rank,
                                             dim=1)

            elif re.match(r'^model.single_blocks.*.linear1.weight$', k):
                qkv, mlp = torch.split(v, [3 * args.hidden_size, mlp_hidden_dim], dim=0)
                split_qkv = split_qkv_tp(qkv, args.num_heads, args.hidden_size,
                                          tensor_parallel, mapping.tp_rank)
                split_mlp = split(mlp, tensor_parallel, mapping.tp_rank, dim=0)
                weights[k] = torch.concat([split_qkv, split_mlp], dim=0)
            elif re.match(r'^model.single_blocks.*.linear1.bias', k):
                qkv_bias, mlp_bias = torch.split(v, [3 * args.hidden_size, mlp_hidden_dim], dim=0)
                split_qkv_bias = split_qkv_bias_tp(qkv_bias, args.num_heads, args.hidden_size,
                                               tensor_parallel, mapping.tp_rank)
                split_mlp_bias = split(mlp_bias, tensor_parallel, mapping.tp_rank, dim=0)
                weights[k] = torch.concat([split_qkv_bias, split_mlp_bias], dim=0)
            elif re.match(r'^model.single_blocks.*.linear2.weight$', k):
                qkv, mlp = torch.split(v, [args.hidden_size, mlp_hidden_dim], dim=1)
                split_qkv = split_matrix_tp(qkv,
                                             tensor_parallel,
                                             mapping.tp_rank,
                                             dim=1)
                split_mlp = split_matrix_tp(mlp,
                                             tensor_parallel,
                                             mapping.tp_rank,
                                             dim=1)
                weights[k] = torch.concat([split_qkv, split_mlp], dim=1)

            elif re.match(r'^model.single_blocks.*.modulation.linear.weight$', k):
                weights[k] = split_matrix_tp(v,
                                             tensor_parallel,
                                             mapping.tp_rank,
                                             dim=0)
            elif re.match(r'^model.single_blocks.*.modulation.linear.bias$', k):
                weights[k] = split(v, tensor_parallel, mapping.tp_rank)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.profiler.print_memory_usage(f'After weights loaded. Total time: {t}', 'MiB')
    return weights

def save_config(args, architecture, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = {
        'architecture': architecture,
        'dtype': args.dtype,
        'num_attention_heads': 24,
        'num_hidden_layers': 60,
        'input_h': args.input_h,
        'input_w': args.input_w,
        'frames': args.frames,
        'in_channels': 16,
        'vec_in_dim': 768,
        'context_in_dim': 4096,
        'hidden_size': args.hidden_size,
        'mlp_ratio': 4.0,
        'num_heads': args.num_heads,
        'depth_double_blocks': 20,
        'depth_single_blocks': 40,
        'rope_dim_list': [16,56,56],
        'theta': 10000,
        'qkv_bias': True,
        'guidance_embed': args.guidance_embed,
        'cp_use_pad': args.cp_use_pad,
        'attn_mode': args.attn_mode,
        'mapping': {
            'world_size': args.cp_size * args.tp_size * args.pp_size,
            'cp_size': args.cp_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        }
    }

    if args.fp8_linear:
        config['quantization'] = {
            'quant_algo': "FP8",
            # TODO: add support for exclude modules.
            'exclude_modules': ["model.final_layer*",
                                "model.txt_in*",
                                "model.time_in*",
                                "model.vector_in*",
                                "model.guidance_in*",
                                "model.double_blocks.0*",
                                "model.single_blocks.39*",
                                ]
        }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    print(config)

def covert_and_save(args, rank):
    uncache_output_dir = args.output_dir+"_uncache"
    cache_output_dir = args.output_dir+"_cache"
    if rank == 0:
        save_config(args, "AriesModelUncache", uncache_output_dir)
        save_config(args, "AriesModelCache", cache_output_dir)

    mapping = Mapping(world_size=args.cp_size * args.tp_size * args.pp_size,
                      rank=rank,
                      cp_size=args.cp_size,
                      tp_size=args.tp_size,
                      pp_size=args.pp_size)

    weights_uncache = convert_video_flux(args, mapping, uncache_output_dir, dtype=args.dtype)

    tensorrt_llm.profiler.print_memory_usage(f'Before uncache weights save.', 'MiB')
    tik = time.time()
    safetensors.torch.save_file(
        weights_uncache, os.path.join(uncache_output_dir, f'rank{rank}.safetensors'))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.profiler.print_memory_usage(f'After Uncache weights save. Total time: {t}', 'MiB')

    weights_cache = convert_video_flux(args, mapping, cache_output_dir, dtype=args.dtype)
    tensorrt_llm.profiler.print_memory_usage(f'Before cache weights save.', 'MiB')
    tik = time.time()
    safetensors.torch.save_file(
        weights_cache, os.path.join(cache_output_dir, f'rank{rank}.safetensors'))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.profiler.print_memory_usage(f'After cache weights save. Total time: {t}', 'MiB')



def execute(workers, func, args):
    if workers == 1:
        for rank, f in enumerate(func):
            f(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [p.submit(f, args, rank) for rank, f in enumerate(func)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(
                exceptions
            ) == 0, "Checkpoint conversion failed, please check error log."

def main():
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.cp_size * args.tp_size * args.pp_size
    if args.workers is None:
        args.workers = min(world_size, torch.cuda.device_count())

    tensorrt_llm.logger.set_level(args.log_level)
    tik = time.time()

    if args.flux_ckpt is None:
        return

    execute(args.workers, [covert_and_save] * world_size, args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
