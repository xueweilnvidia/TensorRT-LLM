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
import math
from typing import List, Optional, Union

import numpy as np
import torch

from .._utils import set_obj_attrs, str_dtype_to_torch, trt_dtype_to_np
from ..functional import (Tensor, arange, concat, constant, cos, embedding, exp,
                          repeat_interleave, select, sin, unsqueeze, where)

from ..mapping import Mapping
from ..module import Module
from ..parameter import Parameter

from .activation import get_activation
from .linear import ColumnLinear, Linear, RowLinear



class Embedding(Module):
    """
    The embedding layer takes input indices (x) and the embedding lookup table (weight) as input.
    And output the corresponding embeddings according to input indices.
    The size of weight is [num_embeddings, embedding_dim]

    Four parameters (tp_size, tp_group, sharding_dim, tp_rank) are involved in tensor parallelism.
    Only when "tp_size > 1 and tp_group is not None", tensor parallelism is enabled.
        When "sharding_dim == 0", the weight is shared in the vocabulary dimension.
            tp_rank must be set when sharding_dim == 0.
        When "sharding_dim == 1",  the weight is shard in the hidden dimension.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 dtype: Optional[str] = None,
                 tp_size: int = 1,
                 tp_group: Optional[list] = None,
                 sharding_dim: int = 0,
                 tp_rank: Optional[int] = None):
        super().__init__()
        # num_embeddings records the total vocab size no matter using TP or not
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.sharding_dim = sharding_dim
        self.tp_rank = tp_rank
        self.dtype = dtype
        self.tp_dim = sharding_dim

        if sharding_dim == 1:
            shape = (self.num_embeddings, self.embedding_dim // self.tp_size)
        elif sharding_dim == 0:
            shape = (math.ceil(self.num_embeddings / self.tp_size),
                     self.embedding_dim)

        self.weight = Parameter(shape=shape, dtype=dtype)

        self.weight_padding_size = ((8 - shape[0] % 8) % 8, shape[1])

        set_obj_attrs(self.weight, {
            "weight_loader": self.weight_loader,
        })

    def forward(self, x):
        # The embedding weight is padded to the multiple of 8.
        # The reason is that when lm_head and vocab_embedding are using the same embedding weight,
        # previously weights can't be depulicated in the engine because gemm will pad the weight to the multiple of 8.
        # If we also pad the embedding weight to the multiple of 8, the weights can be successfully deduplicated.
        # This will not affect the input and output of the gather op and perf impact is negligible.
        if self.weight_padding_size[0] != 0:
            padding_values = np.zeros(self.weight_padding_size,
                                      dtype=trt_dtype_to_np(
                                          self.weight.value.dtype))
            padding = constant(padding_values)
            padded_weight = concat([self.weight.value, padding], dim=0)
        else:
            padded_weight = self.weight.value

        return embedding(x,
                         padded_weight,
                         tp_size=self.tp_size,
                         tp_group=self.tp_group,
                         sharding_dim=self.sharding_dim,
                         tp_rank=self.tp_rank)

    def weight_loader(self, mapping: Mapping, param: Parameter,
                      loaded_weight: torch.Tensor):
        # use_parallel_embedding
        tp_rank = mapping.tp_rank
        if self.tp_size > 1:
            sharding_dim = self.sharding_dim
            shard_size = param._shape[sharding_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(sharding_dim, start_idx,
                                                 shard_size)
        param.value = loaded_weight

    def postprocess(self, tllm_key, weights, **kwargs):
        if weights is None:
            return {}
        weights = weights.to(str_dtype_to_torch(self.dtype))
        return {tllm_key: weights}


class PromptTuningEmbedding(Embedding):
    """
    PromptTuningEmbedding handles fine-tuned prompts with virtual tokens. At runtime,
    a supplementary embedding dictionary is passed. Tokens whose ids are >= vocab_size are embedded
    with that additional dictionary.
    The prompt tuning dictionary holds multiple tasks, and each sequence is assigned a given task.
    Prompt-tuned tokens from a given sequence use the adequate task dictionary, as defined by the `tasks` input.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 vocab_size=None,
                 dtype=None,
                 tp_size=1,
                 tp_group=None,
                 sharding_dim=0,
                 tp_rank=0):
        super().__init__(num_embeddings, embedding_dim, dtype, tp_size,
                         tp_group, sharding_dim, tp_rank)
        if vocab_size is None:
            vocab_size = num_embeddings
        self.vocab_size = vocab_size

    def forward(self, tokens, prompt_embedding_table, tasks, task_vocab_size):
        """
            Pass all tokens through both normal and prompt embedding tables.
            Tokens are masked so that "normal" embedding only see "normal" tokens. Same logic for "prompt" embedding.
            After those two embedding, combine results based on whether the token was "normal" or "prompt-tuned".

        Parameters:
            tokens : Tensor
                the ids to embed, size [batch_size, seq_len]

            prompt_embedding_table : Tensor
                the additional embedding table for prompt-tuned tokens, size [num_tasks * num_tokens_per_task, hidden_size]

            tasks: Tensor
                the task required by each token, size [batch_size, seq_len]

            task_vocab_size: Tensor
                the number of tokens used for each task, should be equal to prompt_embedding_table's num_tokens_per_task, size [1]

        Returns:
            Tokens' embedding
        """
        # do not use ">=" because internally the layer works with floating points
        prompt_tokens_mask = tokens > (self.vocab_size - 1)

        # clip tokens in the [0, vocab_size) range
        normal_tokens = where(prompt_tokens_mask, self.vocab_size - 1, tokens)
        normal_embeddings = embedding(normal_tokens, self.weight.value,
                                      self.tp_size, self.tp_group,
                                      self.sharding_dim, self.tp_rank)

        # put virtual tokens in the [0, max_prompt_vocab_size) range
        prompt_tokens = where(prompt_tokens_mask, tokens - self.vocab_size, 0)

        # add offsets to match the concatenated embedding tables
        tasks = tasks * task_vocab_size

        # tasks: [batch_size, seq_len]
        # prompt_tokens: [batch_size, seq_len]
        prompt_tokens = prompt_tokens + tasks
        prompt_embeddings = embedding(prompt_tokens, prompt_embedding_table)

        # prompt_tokens_mask: [batch_size, seq_len] -> [batch_size, seq_len, 1]
        # combine the correct sources of embedding: normal/prompt
        return where(unsqueeze(prompt_tokens_mask, -1), prompt_embeddings,
                     normal_embeddings)

def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype="float32",
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`float32` or `float64`, *optional*, defaults to `float32`):
            the dtype of the frequency tensor.
    Returns:
        `Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = arange(0, pos, dtype='float32')
    if isinstance(pos, np.ndarray):
        pos = constant(pos)

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (theta ** (np.arange(0, dim, 2, dtype=freqs_dtype)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]
    freqs = constant(freqs)
    freqs = unsqueeze(pos, 1) * unsqueeze(freqs, 0) # [S, D/2]
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = repeat_interleave(cos(freqs), repeats=2, dim=1).cast('float32')  # [S, D]
        freqs_sin = repeat_interleave(sin(freqs), repeats=2, dim=1).cast('float32')  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio
        freqs_cos = concat([cos(freqs), cos(freqs)], dim=-1).cast('float32')  # [S, D]
        freqs_sin = concat([sin(freqs), sin(freqs)], dim=-1).cast('float32')  # [S, D]
        return freqs_cos, freqs_sin
    else:
        raise NotImplementedError()

class FluxPosEmbed(Module):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.cast("float32")
        # TODO: TRT don't support float64, so we use float32 here, this might lead to accuracy issue
        freqs_dtype = "float32" # "float64"
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i], select(pos, dim=-1, index=i), repeat_interleave_real=True, use_real=True, freqs_dtype=freqs_dtype
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = concat(cos_out, dim=-1)
        freqs_sin = concat(sin_out, dim=-1)
        return freqs_cos, freqs_sin


def get_timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * np.arange(
        start=0, stop=half_dim, dtype=np.float32
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    exponent = constant(exponent)

    emb = exp(exponent)
    emb = unsqueeze(timesteps, -1).cast('float32') * unsqueeze(emb, 0)

    # scale embeddings
    emb = scale * emb

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = concat([cos(emb), sin(emb)], dim=-1)
    else:
        emb = concat([sin(emb), cos(emb)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        raise NotImplementedError()
    return emb

class TimestepEmbedding(Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
        mapping=None,
        dtype=None
    ):
        super().__init__()
        tp_group = mapping.tp_group
        tp_size = mapping.tp_size
        self.linear_1 = ColumnLinear(in_channels, time_embed_dim, sample_proj_bias, tp_group=tp_group, tp_size=tp_size, dtype=dtype, gather_output=False)

        if cond_proj_dim is not None:
            self.cond_proj = Linear(cond_proj_dim, in_channels, bias=False, dtype=dtype)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = RowLinear(time_embed_dim, time_embed_dim_out, sample_proj_bias, tp_group=tp_group, tp_size=tp_size, dtype=dtype)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps) -> Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb

class PixArtAlphaTextProjection(Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh", mapping=None, dtype=None):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        tp_group = mapping.tp_group
        tp_size = mapping.tp_size
        self.linear_1 = ColumnLinear(in_features=in_features, out_features=hidden_size, bias=True, tp_group=tp_group, tp_size=tp_size, dtype=dtype, gather_output=False)
        self.act_1 = get_activation(act_fn)
        self.linear_2 = RowLinear(in_features=hidden_size, out_features=out_features, bias=True, tp_group=tp_group, tp_size=tp_size, dtype=dtype)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class CombinedTimestepGuidanceTextProjEmbeddings(Module):
    def __init__(self, embedding_dim, pooled_projection_dim, mapping=None, dtype=None):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, mapping=mapping, dtype=dtype)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, mapping=mapping, dtype=dtype)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu", mapping=mapping, dtype=dtype)

    def forward(self, timestep, guidance, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.cast(pooled_projection.dtype))  # (N, D)

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.cast(pooled_projection.dtype))  # (N, D)

        time_guidance_emb = timesteps_emb + guidance_emb

        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = time_guidance_emb + pooled_projections

        return conditioning

class CombinedTimestepTextProjEmbeddings(Module):
    def __init__(self, embedding_dim, pooled_projection_dim, mapping=None, dtype=None):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, mapping=mapping, dtype=dtype)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu", mapping=mapping, dtype=dtype)

    def forward(self, timestep: Tensor, pooled_projection: Tensor):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.cast(dtype=pooled_projection.dtype))  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projections

        return conditioning
