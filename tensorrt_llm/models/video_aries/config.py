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
from typing import Optional, Union

import torch

from ..._utils import torch_dtype_to_str
from ...mapping import Mapping
from ..modeling_utils import PretrainedConfig, QuantConfig


class AriesParamsConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 in_channels: int = 64,
                 out_channels: int = None,  
                 vec_in_dim: int = 768,
                 context_in_dim: int = 4096, 
                 hidden_size: int = 3072, 
                 mlp_ratio: float = 4.0, 
                 num_heads: int = 24,   
                 depth_double_blocks: int = 19,
                 depth_single_blocks: int = 38,
                 rope_dim_list: list[int] = [16, 56, 56], 
                 theta: int = 10000,
                 qkv_bias: bool = True,          
                 guidance_embed: bool = False,   
                 reverse: bool = True,          
                 input_h: int = 192,
                 input_w: int = 396,
                 **kwargs):

        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.vec_in_dim = vec_in_dim
        self.context_in_dim = context_in_dim
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.depth_double_blocks = depth_double_blocks
        self.depth_single_blocks = depth_single_blocks
        self.rope_dim_list = rope_dim_list
        self.theta = theta
        self.qkv_bias = qkv_bias
        self.guidance_embed = guidance_embed
        self.reverse = reverse
        self.input_h = input_h
        self.input_w = input_w
        self.attn_mode = "plugin" #plugin or ootb


    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in QWenConfig
        output['input_h'] = self.input_h
        output['input_w'] = self.input_w
        output['in_channels'] = self.in_channels
        output['vec_in_dim'] = self.vec_in_dim
        output['context_in_dim'] = self.context_in_dim
        output['hidden_size'] = self.hidden_size
        output['mlp_ratio'] = self.mlp_ratio
        output['num_heads'] = self.num_heads
        output['depth_double_blocks'] = self.depth_double_blocks
        output['depth_single_blocks'] = self.depth_single_blocks
        output['rope_dim_list'] = self.rope_dim_list
        output['theta'] = self.theta
        output['qkv_bias'] = self.qkv_bias
        output['guidance_embed'] = self.guidance_embed
        output['reverse'] = self.reverse
        return output


