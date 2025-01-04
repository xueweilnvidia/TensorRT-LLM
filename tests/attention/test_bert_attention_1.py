# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import unittest
from itertools import product

import numpy as np
import torch
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner
# from transformers import BertConfig
# from transformers.models.bert.modeling_bert import BertSelfAttention

import flash_attn
from flash_attn.flash_attn_interface import _flash_attn_forward
from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_func

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.plugin.plugin import ContextFMHAType

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import skip_fp32_accum_pre_ampere, unittest_name_func



def test_bert_attention(batch_size, in_len, num_heads, head_size,
                        context_fmha_type, dtype):
    skip_fp32_accum_pre_ampere(context_fmha_type)

    def _construct_execution(input_tensor, input_lengths,
                                num_heads, hidden_size, output, dtype,
                                shape_dict):
        head_size = hidden_size // num_heads
        # construct trt network
        builder = tensorrt_llm.Builder()
        builder.strongly_typed = False  # Test need to run in weekly typed mode
        net = builder.create_network()
        net.plugin_config.to_legacy_setting()
        net.plugin_config.bert_attention_plugin = dtype
        net.plugin_config.set_context_fmha(context_fmha_type)
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            qkv = Tensor(name='input',
                                shape=tuple(input_tensor.shape),
                                dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            input_lengths_tensor = Tensor(
                name='input_lengths',
                shape=tuple(input_lengths.shape),
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))


            # attention (padding mask)
            outputs = tensorrt_llm.functional.bert_attention(
                qkv,
                input_lengths_tensor,
                num_heads=num_heads,
                head_size=head_size,
                q_scaling=1.0)

            network.mark_output(outputs.trt_tensor)
            outputs.trt_tensor.name = 'output'
            outputs.trt_tensor.dtype = tensorrt_llm.str_dtype_to_trt(dtype)

        engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(bf16=(dtype == 'bfloat16')))

        with TrtRunner(engine) as runner:
            outputs = runner.infer(feed_dict={
                'input': input_tensor,
                'input_lengths': input_lengths
            })

        return outputs['output']

    hidden_size = num_heads * head_size
    shape_dict = {
        'weight': (hidden_size, hidden_size * 3),
        'bias': (hidden_size * 3, ),
        'input_lengths': (batch_size, ),
    }
    

    input_lengths = torch.ones(
        (batch_size, ), dtype=torch.int32, device='cuda') * in_len
    # input_lengths[0] = 5120
    # input_lengths[1] = 20

    # Context stage
    shape_dict['input'] = (batch_size, in_len, 3, hidden_size)
    shape_dict['output'] = (batch_size, in_len, 1, hidden_size)

    input_tensor = torch.randn(
        shape_dict['input'],
        dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
        device='cuda')
    input_tensor[:, 0:32, :, :] = input_tensor[:, 0:32, :, :] * 5
    input_tensor[:, 128:160, :, :] = input_tensor[:, 128:160, :, :] * 10

    output = torch.zeros(
        shape_dict['output'],
        dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
        device='cuda')
    output = _construct_execution(input_tensor, input_lengths,
                                    num_heads, hidden_size, output, dtype,
                                    shape_dict)

    output = _construct_execution(input_tensor, input_lengths,
                                    num_heads, hidden_size, output, dtype,
                                    shape_dict)
    
    output = _construct_execution(input_tensor, input_lengths,
                                    num_heads, hidden_size, output, dtype,
                                    shape_dict)

    q = input_tensor[:,:,0,:].squeeze().reshape(batch_size, in_len, num_heads, head_size)
    k = input_tensor[:,:,1,:].squeeze().reshape(batch_size, in_len, num_heads, head_size)
    v = input_tensor[:,:,2,:].squeeze().reshape(batch_size, in_len, num_heads, head_size)

    attn_output1 = flash_attn_func(q, k, v)

    output = output
    print (output)
    print (attn_output1)

    attn_output = output.to(torch.float32).reshape(-1).to("cuda")
    attn_output1 = attn_output1.to(torch.float32).reshape(-1)
 
    a = torch.norm(attn_output)
    b = torch.norm(attn_output1)
    print(a)
    print(b)

    ab = torch.sum(torch.abs(torch.mul(attn_output1, attn_output)))
    print(ab)


    # output = cos_sim(attn_output.reshape(-1), attn_output1.reshape(-1))

    print("finished: ", ab/(a*b))

    



if __name__ == "__main__":
    test_bert_attention(1, 118800, 3, 128, ContextFMHAType.enabled_with_fp32_acc, "bfloat16")
