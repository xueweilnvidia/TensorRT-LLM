from dataclasses import dataclass
from typing import Tuple, Union, Optional

import math
import numpy as np
import tensorrt as trt
from ..._common import default_net
from ..._utils import (bf16_array, bool_array, dim_resolve_negative,
                     dim_to_trt_axes, dims_array, fp16_array, fp32_array,
                     int32_array, int64_array, np_dtype_to_trt,
                     str_dtype_to_trt, trt_dtype_to_np, trt_dtype_to_str)
from ...functional import (Tensor, allgather, arange, chunk, concat, stack, cast,constant,sum, 
                           cos, exp, expand, repeat_interleave, shape, silu, gelu, sin, slice,where,
                           split, dynamic_split, select, index_select, gather, einsum, outer, pow, identity, tanh,
                           unsqueeze, squeeze, matmul, softmax, activation, bert_attention)
from ...layers import MLP, Conv2d,Conv3d, Embedding, LayerNorm, RmsNorm, Linear, RowLinear
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...quantization import QuantMode

class PatchEmbed(Module):
    def __init__(                                                                   
            self,                                                                   
            patch_size,                                                          
            in_chans=3,                                                             
            embed_dim=768,                                                          
            flatten=True,                                                           
            bias=True,                                                              
            dtype: trt.DataType =None,                                                             
    ):                                                                              
        factory_kwargs = {'dtype': dtype}                         
        super().__init__()                                                          
        self.patch_size = patch_size                                                
        self.flatten = flatten                                                      
                                                                                    
        self.proj = Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias,                                                                
                              **factory_kwargs)                                     
                                                                                    
    def forward(self, x):                                                           
        x = self.proj(x)                                                            
        if self.flatten:                                                            
            #x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC                         
            B = shape(x,0)
            C = shape(x,1)
            F = shape(x,2)
            H = shape(x,3)
            W = shape(x,4)
            x = x.view(concat([B,C,F*H*W])).transpose(1, 2)  # BCHW -> BNC                         

        return x                                                                    

 
def reshape_for_broadcast(freqs_cis: Union[Tensor, Tuple[Tensor]], x: Tensor, head_first=False):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Notes:
        When using FlashMHAModified, head_first should be False.
        When using Attention, head_first should be True.

    Args:
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim()
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (x.shape[-2], x.shape[-1]), f'freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}'
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (x.shape[1], x.shape[-1]), f'freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}'
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(shape), freqs_cis[1].view(shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == [x.shape[-2], x.shape[-1]], f'freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}'
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis.shape == [x.shape[1], x.shape[-1]], f'freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}'
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(shape)


def rotate_half(x):
    head_dim = x.shape[-1]
    ndim = x.ndim()
    old_shape = shape(x)
    #x_real, x_imag = x.cast(trt.float32).view([*x.shape[:-1], head_dim//2, 2]).unbind(-1)  # [B, S, H, D//2]
    new_shape = concat([shape(x,i) for i in range(ndim-1)])
    x_real, x_imag = x.cast(trt.float32).view(concat([new_shape, head_dim//2, 2])).unbind(-1)  # [B, S, H, D//2]
    zero = constant(fp32_array(0.0))
    #zero = constant(np.ascontiguousarray(np.zeros([1], dtype=np.float32)))

    #return stack([zero-x_imag, x_real], dim=-1).flatten(3)
    return stack([zero-x_imag, x_real], dim=-1).view(old_shape)


def apply_rotary_emb(
        xq: Tensor,
        xk: Tensor,
        freqs_cis: Union[Tensor, Tuple[Tensor, Tensor]],
        head_first: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (torch.Tensor or tuple): Precomputed frequency tensor for complex exponential.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)    # [S, D]
        xq_out = (xq.cast(trt.float32) * cos + rotate_half(xq.cast(trt.float32)) * sin).cast(xq.dtype)
        xk_out = (xk.cast(trt.float32) * cos + rotate_half(xk.cast(trt.float32)) * sin).cast(xk.dtype)
    else:
        assert isinstance(freqs_cis, tuple), "apply_rotary_emb only support freqs_cis is tuple case"

    return xq_out, xk_out



def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    exponent = [
        i * -math.log(max_period) / half
        for i in range(half)
    ]
    freqs = exp(constant(fp32_array(exponent)))
    #freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)

    #args = t[:, None].float() * freqs[None]
    args = unsqueeze(t, -1).cast(trt.float32) * unsqueeze(freqs, 0)
    embedding = concat([cos(args), sin(args)], dim=-1)
    if dim % 2:
        indices = constant(int32_array([0]))
        embedding_sel = index_select(embedding, dim=1, index=indices)
        zero = constant(
            np.ascontiguousarray(
                np.zeros([embedding_sel.shape], dtype=trt_dtype_to_np(embedding.dtype))))
        #embedding = concat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        embedding = concat([embedding, zero], dim=-1)
    def is_floating_point(t):
        dtype = t.dtype
        return dtype in [trt.float32, trt.float16, trt.bfloat16]
    if is_floating_point(t):
        embedding = embedding.cast(t.dtype)
    return embedding


class MLPEmbedder(Module):
    def __init__(self, in_dim: int, hidden_dim: int, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.in_layer = Linear(in_dim, hidden_dim, bias=True, dtype=dtype)
        self.out_layer = Linear(hidden_dim, hidden_dim, bias=True, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(silu(self.in_layer(x)))



@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(Module):
    def __init__(self, dim: int, double: bool, dtype=None):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = Linear(dim, self.multiplier * dim, bias=True, dtype=dtype)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = chunk(unsqueeze(self.lin(silu(vec)),1),self.multiplier, dim=-1)
        #out = self.lin(silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        #out0, out1 = split(out, [3, out.shape[0]-3], dim=0)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

class ModulateDiT(Module):
    def __init__(self, hidden_size: int, factor: int, act_layer: callable, mapping=Mapping(), dtype: trt.DataType=None):
        factory_kwargs = {'dtype': dtype}
        super().__init__()
        self.act = act_layer
        self.linear = Linear(hidden_size, factor * hidden_size, tp_group=mapping.tp_group, tp_size=mapping.tp_size, bias=True, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.act(x))

def modulate(x, shift=None, scale=None, dtype: trt.DataType=None): 
    ones = 1.0
    if dtype is not None:
        ones = constant(np.ones(1, dtype=np.float32)).cast(dtype)
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (ones + unsqueeze(scale,1))
    elif scale is None:
        return x + unsqueeze(shift,1)
    else:
        return x * (ones + unsqueeze(scale,1)) + unsqueeze(shift, 1)  

def apply_gate(x, gate=None, tanh=False):
    if gate is None:
        return x
    if tanh:
        return x * tanh(unsqueeze(gate,1))
    else:
        return x * unsqueeze(gate,1)

class LastLayer(Module):
    def __init__(self, hidden_size, patch_size, out_channels, mapping=Mapping(), dtype: trt.DataType=None):
        factory_kwargs = {'dtype': dtype}
        super().__init__()
        self.dtype = dtype
        self.mapping = mapping
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        if isinstance(patch_size, int):
            self.linear = Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype)
        else:
            self.linear = Linear(hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=True, dtype=dtype)
        #self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.adaLN_modulation = Linear(hidden_size, 2 * hidden_size, 
                                       tp_group=mapping.tp_group,
                                       tp_size=mapping.tp_size,
                                       bias=True, dtype=dtype)


    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = chunk(self.adaLN_modulation(silu(vec)), 2, dim=1)
        #x = modulate(self.norm_final(x), shift=shift, scale=scale)
        ones = constant(np.ones(1, dtype=np.float32)).cast(self.dtype)
        x = (ones + unsqueeze(scale, 1)) * self.norm_final(x) + unsqueeze(shift, 1)
        x = self.linear(x)
        return x

class TextProjection(Module):
    """
    Projects text embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_channels, hidden_size, act_layer, dtype: trt.DataType=None):
        factory_kwargs = {'dtype': dtype}
        super().__init__()
        self.linear_1 = Linear(in_features=in_channels, out_features=hidden_size, bias=True, **factory_kwargs)
        self.act_1 = act_layer
        self.linear_2 = Linear(in_features=hidden_size, out_features=hidden_size, bias=True, **factory_kwargs)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class TimestepEmbedder(Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self,
                 hidden_size,
                 act_layer=silu,
                 frequency_embedding_size=256,
                 max_period=10000,
                 out_size=None,
                 dtype: trt.DataType=None,
                 ):
        factory_kwargs = {'dtype': dtype}
        super().__init__()
        self.dtype = dtype
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if isinstance(act_layer, str):
            if act_layer == 'silu':
                self.act = silu
        else:
            self.act = act_layer
        if out_size is None:
            out_size = hidden_size

        self.mlp1 = Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs)
        self.mlp2 = Linear(hidden_size, out_size, bias=True, **factory_kwargs)

    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = exp(
            -math.log(max_period) *
            arange(start=0, end=half, dtype=trt_dtype_to_str(trt.float32)) /
            constant(np.array([half], dtype=np.float32)))
        args = unsqueeze(t, -1).cast(trt.float32) * unsqueeze(freqs, 0)
        embedding = concat([cos(args), sin(args)], dim=-1)
        if self.dtype is not None: embedding = embedding.cast(self.dtype)
        assert dim % 2 == 0
        return embedding

    def forward(self, t):
        #t_freq = self.timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        t_freq = self.mlp1(t_freq)
        if self.act is not None:
            t_freq = self.act(t_freq)
        t_emb = self.mlp2(t_freq)
        return t_emb


class TRTAttention(Module):
    def __init__(self, num_attention_heads, attention_head_size, q_scaling=1.0, use_plugin=False, mapping=Mapping(), layer_name=None):
        super().__init__()
        self.use_plugin = use_plugin
        self.tp_group = mapping.tp_group
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.cp_size = mapping.cp_size
        self.cp_group = mapping.cp_group

        self.attention_head_size = attention_head_size
        self.num_attention_heads = num_attention_heads // self.tp_size
        self.attention_hidden_size = self.attention_head_size * self.num_attention_heads
        self.q_scaling = q_scaling
        norm_factor = math.sqrt(self.attention_head_size)
        self.score_factor = 1 / (q_scaling * norm_factor)
        self.layer_name = layer_name

    def forward(self, query, key, value, attention_mask=None, input_lengths=None, max_input_length=None):
        """
        input format: 'bshd, output bs(hd)'
        """
        if self.use_plugin: 
            #assert default_net().plugin_config.bert_attention_plugin
            assert self.cp_size == 1, "BertAttentionPlugin don't support cp now."
            b, s = shape(query, 0), shape(query, 1)
            # 'bshd' to 'bs(hd)'
            query = query.view(concat([b, s, self.num_attention_heads * self.attention_head_size]))
            key = key.view(concat([b, s, self.num_attention_heads * self.attention_head_size]))
            value = value.view(concat([b, s, self.num_attention_heads * self.attention_head_size]))
            qkv = concat([query, key, value], dim=-1)
            #'bs(3hd)'

            if input_lengths is None:
                input_lengths = repeat_interleave(unsqueeze(s, 0), repeats=b, dim=0).cast(trt.DataType.INT32)
            if self.layer_name is not None:
                self.register_network_output(self.layer_name+"_qkv_debug_output", qkv)
                self.register_network_output(self.layer_name+"_input_lengths_debug_output", input_lengths)
            context = bert_attention(qkv, input_lengths, self.num_attention_heads, self.attention_head_size, q_scaling=self.q_scaling, max_input_length=max_input_length)
            if self.layer_name is not None:
                self.register_network_output(self.layer_name+"_bert_atn_debug_output", context)
            # 'bs(hd)'
            return context
        else:
            if self.cp_size > 1 and self.cp_group is not None:
                key = allgather(key, self.cp_group, gather_dim=1)
                value = allgather(value, self.cp_group, gather_dim=1)
            return self.trt_ootb_attention(query, key, value, attention_mask)


    def trt_ootb_attention(self, query, key, value, attention_mask):
        """
        input format: 'bshd'
        """
        def transpose_for_scores(x):
            new_x_shape = concat([
                shape(x, 0),
                shape(x, 1), self.num_attention_heads,
                self.attention_head_size
            ])
            #return x.view(new_x_shape).permute([2, 1, 0, 3]) # sbhd -> hbsd
            #return x.view(new_x_shape).permute([2, 0, 1, 3]) # bshd -> hbsd
            return x.view(new_x_shape).permute([0, 2, 1, 3]) # bshd -> bhsd

        query = transpose_for_scores(query)
        key = transpose_for_scores(key)
        value = transpose_for_scores(value)

        key = key.permute([0, 1, 3, 2]) # bhds
        attention_scores = matmul(query, key, use_fp32_acc=False)
        attention_scores = attention_scores * self.score_factor
        if attention_mask is not None:
            if attention_mask.dtype == trt.bool:
                attention_mask = where(attention_mask == 0, float('-inf'), 0.0)
                attention_mask = repeat_interleave(attention_mask, self.num_attention_heads, dim=1)
            attention_mask = cast(attention_mask, attention_scores.dtype)
            attention_scores += attention_mask

        attention_probs = softmax(attention_scores, dim=-1)

        context = matmul(attention_probs, value,
                            use_fp32_acc=False).permute([0, 2, 1, 3]) # bhsd->bshd
        context = context.view(
            concat([
                shape(context, 0),
                shape(context, 1), self.attention_hidden_size
            ]))
        return context




class IndividualTokenRefinerBlock(Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        mapping=Mapping(),
        dtype: Optional[trt.DataType] = None,
        quant_mode = QuantMode(0),
    ):
        factory_kwargs = {'dtype': dtype}
        super().__init__()
        import copy
        self.mapping = copy.deepcopy(mapping)
        self.mapping.cp_size = 1
        self.mapping.cp_group = 1

        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.norm1 = LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.self_attn_qkv = Linear(hidden_size, hidden_size * 3, bias=qkv_bias, tp_group=mapping.tp_group, tp_size=mapping.tp_size, gather_output=False, is_qkv=True, **factory_kwargs)
        if qk_norm_type == "layer":
            q_norm_layer = LayerNorm(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            k_norm_layer = LayerNorm(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.self_attn_q_norm = (
            q_norm_layer
            if qk_norm
            else identity
        )
        self.self_attn_k_norm = (
            k_norm_layer
            if qk_norm
            else identity
        )
        self.self_attn_proj = RowLinear(hidden_size, hidden_size, bias=qkv_bias,tp_group=mapping.tp_group, tp_size=mapping.tp_size, **factory_kwargs)
        self.core_attention = TRTAttention(
            num_attention_heads=num_heads,
            attention_head_size=head_dim,
            q_scaling=1.0, 
            use_plugin=False,
            mapping=self.mapping)

        self.norm2 = LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.act_type = None
        if act_type == "silu":
            self.act_type = silu
        self.mlp = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=mlp_hidden_dim,
            hidden_act=act_type,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,  
            quant_mode = quant_mode,
            **factory_kwargs,
        )

        self.adaLN_modulation = Linear(hidden_size, 2 * hidden_size, tp_group=mapping.tp_group, tp_size=mapping.tp_size, bias=True, **factory_kwargs)

    def forward(
        self,
        x: Tensor,
        c: Tensor, # timestep_aware_representations + context_aware_representations
        attn_mask: Tensor = None,
    ):
        gate_msa, gate_mlp = chunk(self.adaLN_modulation(self.act_type(c)), 2, dim=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        #q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        B,L,_ = qkv.shape
        q, k, v = qkv.view([B, L, 3, self.num_heads//self.mapping.tp_size, self.head_dim]).permute([2,0,1,3,4]).unbind(dim=0)

        # Apply QK-Norm if needed
        q = self.self_attn_q_norm(q)
        k = self.self_attn_k_norm(k)

        # Self-Attention
        #attn = attention(q, k, v, mode="torch", attn_mask=attn_mask)
        attn = self.core_attention(q,k,v,attention_mask=attn_mask)

        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)

        # FFN Layer
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)

        return x


class IndividualTokenRefiner(Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        depth,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        mapping = Mapping(),
        dtype: Optional[trt.DataType] = None,
        quant_mode = QuantMode(0),
    ):
        factory_kwargs = {'dtype': dtype}
        self.mapping = mapping
        super().__init__()
        self.blocks = ModuleList([
            IndividualTokenRefinerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                mlp_drop_rate=mlp_drop_rate,
                act_type=act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
                mapping = mapping,
                quant_mode=quant_mode,
                **factory_kwargs,
            ) for _ in range(depth)
        ])

    def forward(
        self,
        x: Tensor,
        #c: torch.LongTensor,
        c: Tensor,
        mask: Optional[Tensor] = None,
    ):
        self_attn_mask = None
        if mask is not None:
            #batch_size = mask.shape[0]
            seq_len = mask.shape[1]
            # batch_size x 1 x seq_len x seq_len
            #self_attn_mask_1 = mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            self_attn_mask_1 = repeat_interleave(mask.view(concat([shape(mask,0), 1, 1, seq_len])), seq_len, dim=2)
            #self_attn_mask_1 = expand(mask.view([batch_size, 1, 1, seq_len]), [batch_size, 1, seq_len, seq_len],
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            # batch_size x 1 x seq_len x seq_len, 1 for broadcasting of num_heads
            #self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            self_attn_mask = (self_attn_mask_1 * self_attn_mask_2).cast(trt.bool)
            # avoids self-attention weight being NaN for padding tokens
            #self_attn_mask[:, :, :, 0] = True
            first_part, rest_part = self_attn_mask.split([1, seq_len-1], dim=-1)
            true_tensor_c = constant(np.ascontiguousarray(np.full([1, 1, seq_len], True, dtype=bool)))
            true_tensor = repeat_interleave(true_tensor_c, shape(mask,0), dim=0)
            self_attn_mask = concat([unsqueeze(true_tensor, -1), rest_part], dim=-1)

        for block in self.blocks:
            x = block(x, c, self_attn_mask)
        return x


class SingleTokenRefiner(Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        num_heads,
        depth,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        attn_mode: str = "ootb",
        mapping=Mapping(),
        dtype: Optional[trt.DataType] = None,
        quant_mode = QuantMode(0),
    ):
        factory_kwargs = {'dtype': dtype}
        super().__init__()
        self.attn_mode = attn_mode
        self.dtype = dtype
        self.mapping = mapping
        assert self.attn_mode in ["ootb", "plugin"], "Only support 'ootb' or 'plugin' mode for now."

        self.input_embedder = Linear(in_channels, hidden_size, bias=True, **factory_kwargs)

        if act_type == "silu":
            act_layer_func = silu
        else:
            act_layer_func = None

        # Build timestep embedding layer
        self.t_embedder = TimestepEmbedder(hidden_size, act_layer_func, **factory_kwargs)
        # Build context embedding layer
        self.c_embedder = TextProjection(in_channels, hidden_size, act_layer_func, **factory_kwargs)

        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
            mapping=self.mapping,
            quant_mode=quant_mode,
            **factory_kwargs
        )

    def forward(
        self,
        x: Tensor,
        #t: torch.LongTensor,
        t: Tensor,
        #mask: Optional[torch.LongTensor] = None,
        mask: Optional[Tensor] = None,
    ):
        timestep_aware_representations = self.t_embedder(t)

        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = unsqueeze(mask.cast(trt.float32), -1)   # [b, s1, 1]
            context_aware_representations = (
                    sum(x.cast(trt.float32) * mask_float, 1) / sum(mask_float,1)
                    )
            context_aware_representations=cast(context_aware_representations, self.dtype)
        context_aware_representations = self.c_embedder(context_aware_representations)
        c = timestep_aware_representations + context_aware_representations

        x = self.input_embedder(x)
        
        x = self.individual_token_refiner(x, c, mask)

        return x


class DoubleStreamBlock(Module):
    layer_index = 0
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_act_type: str = 'gelu_tanh',
        qk_norm: bool = True,
        qk_norm_type: str = 'rms',
        qkv_bias: bool = False,
        attn_mode: str = 'ootb',
        reverse: bool = False,
        mapping=Mapping(),
        dtype: Optional[trt.DataType] = None,
        quant_mode = QuantMode(0),
    ):
        factory_kwargs = {'dtype': dtype}
        super().__init__()
        self.layer_index_ = DoubleStreamBlock.layer_index
        DoubleStreamBlock.layer_index += 1
        self.mapping = mapping
        self.dtype = dtype
        self.reverse = reverse
        self.attn_mode = attn_mode
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.head_size = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=silu, mapping=self.mapping, **factory_kwargs)
        self.img_norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.img_attn_qkv = Linear(hidden_size, hidden_size * 3, bias=qkv_bias,tp_group=mapping.tp_group, tp_size=mapping.tp_size, gather_output=False, is_qkv=True,  **factory_kwargs)

        if qk_norm_type == 'rms':
            q_norm_layer = RmsNorm(normalized_shape=head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype)
            k_norm_layer = RmsNorm(normalized_shape=head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype)
            txt_q_norm_layer = RmsNorm(normalized_shape=head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype)
            txt_k_norm_layer = RmsNorm(normalized_shape=head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype)
        else:
            q_norm_layer = None
            k_norm_layer = None
        self.img_attn_q_norm = (
            q_norm_layer
            if qk_norm
            else identity
        )
        self.img_attn_k_norm = (
            k_norm_layer
            if qk_norm
            else identity
        )
        self.img_attn_proj = RowLinear(hidden_size, hidden_size, tp_group=mapping.tp_group, tp_size=mapping.tp_size, bias=qkv_bias, **factory_kwargs)

        self.is_use_plugin = False
        if attn_mode == 'plugin':
            self.is_use_plugin = True
        self.core_attention = TRTAttention(
            num_attention_heads=num_heads,
            attention_head_size=head_dim,
            q_scaling=1.0, 
            use_plugin=self.is_use_plugin,
            mapping=mapping,
            layer_name="DoubleStreamBlock"+str(self.layer_index_))

        self.img_norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=mlp_hidden_dim,
            hidden_act=mlp_act_type, #nn.GELU(approximate='tanh')
            bias=True,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            quant_mode=quant_mode,
            **factory_kwargs,
        )
        self.txt_mod = ModulateDiT(hidden_size, factor=6, act_layer=silu, mapping=self.mapping, **factory_kwargs)
        self.txt_norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.txt_attn_qkv = Linear(hidden_size, hidden_size * 3, bias=qkv_bias, tp_group=mapping.tp_group, tp_size=mapping.tp_size, gather_output=False, is_qkv=True, **factory_kwargs)
        self.txt_attn_q_norm = (
            txt_q_norm_layer
            if qk_norm
            else identity
        )
        self.txt_attn_k_norm = (
            txt_k_norm_layer
            if qk_norm
            else identity
        )
        self.txt_attn_proj = RowLinear(hidden_size, hidden_size, tp_group=mapping.tp_group, tp_size=mapping.tp_size, bias=qkv_bias, **factory_kwargs)

        self.txt_norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=mlp_hidden_dim,
            hidden_act=mlp_act_type, #nn.GELU(approximate='tanh')
            bias=True,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            quant_mode=quant_mode,
            **factory_kwargs,
        )

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        attn_mask: Tensor,
        vec: Tensor,
        attn_input_length: Tensor = None,
        attn_max_input_length: Tensor = None,
        freqs_cis: tuple = None
    ) -> Tuple[Tensor, Tensor]:
        num_heads = self.num_heads // self.mapping.tp_size
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = (
            chunk(self.img_mod(vec),6, dim=-1)
        )
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = (
            chunk(self.txt_mod(vec),6, dim=-1)
        )

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale, dtype=self.dtype)
        img_qkv = self.img_attn_qkv(img_modulated)
        #img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        #B,L,_ = img_qkv.shape
        B = shape(img_qkv,0)
        L = shape(img_qkv,1)
        img_q, img_k, img_v = img_qkv.view(concat([B, L, 3, num_heads, self.head_size])).permute([2,0,1,3,4]).unbind(dim=0)
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q)
        img_k = self.img_attn_k_norm(img_k)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale, dtype=self.dtype)
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        #txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        B,L,_ = txt_qkv.shape
        txt_q, txt_k, txt_v = txt_qkv.view([B, L, 3, num_heads, self.head_size]).permute([2,0,1,3,4]).unbind(dim=0)
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q)
        txt_k = self.txt_attn_k_norm(txt_k)

        # TODO(jarvizhang): Currently only support `torch` mode
        txt_seq_len = shape(txt,1)
        img_seq_len = shape(img,1)
        if self.reverse:
            # Run actual attention.
            q = concat([img_q, txt_q], dim=1)
            k = concat([img_k, txt_k], dim=1)
            v = concat([img_v, txt_v], dim=1)

            #attn = attention(q, k, v, mode=self.attn_mode, attn_mask=attn_mask)
            attn=self.core_attention(q, k, v, attention_mask=attn_mask, input_lengths=attn_input_length, max_input_length=attn_max_input_length)
            #img_attn, txt_attn = attn[:, :img.shape[1]], attn[:, img.shape[1]:]
            #img_attn, txt_attn = split(attn,[img.shape[1], attn.shape[1]-img.shape[1]], dim=1)
            img_attn, txt_attn = dynamic_split(attn,[img_seq_len, shape(attn,1)-img_seq_len], dim=1)
        else:
            # Run actual attention.
            q = concat([txt_q, img_q], dim=1)
            k = concat([txt_k, img_k], dim=1)
            v = concat([txt_v, img_v], dim=1)

            #attn = attention(q, k, v, mode=self.attn_mode, attn_mask=attn_mask)
            attn=self.core_attention(q, k, v, attention_mask=attn_mask)
            #txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]
            txt_attn, img_attn = dynamic_split(attn, [txt_seq_len, shape(attn,1)-txt_seq_len], dim=1)

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)), gate=img_mod2_gate)

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)), gate=txt_mod2_gate)
        
        return img, txt


class SingleStreamBlock(Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """
    layer_index = 0
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        mlp_act_type: str = 'gelu_tanh',
        qk_norm: bool = True,
        qk_norm_type: str = 'rms',
        qk_scale: float = None,
        attn_mode: str = 'ootb',
        reverse: bool = False,
        mapping=Mapping(),
        dtype: Optional[trt.DataType] = None,
        quant_mode = QuantMode(0),
    ):
        factory_kwargs = {'dtype': dtype}
        super().__init__()
        self.layer_index_ = SingleStreamBlock.layer_index
        SingleStreamBlock.layer_index += 1

        self.mapping = mapping
        self.dtype = dtype
        self.reverse = reverse
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.head_size = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5
        self.quant_mode = quant_mode

        self.is_use_plugin = False
        if attn_mode == 'plugin':
            self.is_use_plugin = True
        self.core_attention = TRTAttention(
            num_attention_heads=num_heads,
            attention_head_size=head_dim,
            q_scaling=1.0, 
            use_plugin=self.is_use_plugin,
            mapping=mapping,
            layer_name="SingleStreamBlock"+str(self.layer_index_))
        # qkv and mlp_in
        self.linear1 = Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim, tp_group=mapping.tp_group, tp_size=mapping.tp_size, gather_output=False,  **factory_kwargs)
        # proj and mlp_out
        self.linear2 = RowLinear(hidden_size + mlp_hidden_dim, hidden_size, tp_group=mapping.tp_group, tp_size=mapping.tp_size, **factory_kwargs)

        if qk_norm_type == 'rms':
            q_norm_layer = RmsNorm(normalized_shape=head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            k_norm_layer = RmsNorm(normalized_shape=head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        else:
            q_norm_layer = None
            k_norm_layer = None
        self.q_norm = (
            q_norm_layer
            if qk_norm
            else identity
        )
        self.k_norm = (
            k_norm_layer
            if qk_norm
            else identity
        )

        self.pre_norm = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        #self.mlp_act = get_activation_layer(mlp_act_type)()
        #self.mlp_act = nn.GELU(approximate="tanh")
        if mlp_act_type in ["gelu_tanh", "gelu_pytorch_tanh"]:
            self.mlp_act = gelu
        else:
            self.mlp_act = None
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=silu, mapping=mapping,  **factory_kwargs)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor,
        vec: Tensor,
        txt_len: int,
        attn_input_length: Tensor = None,
        attn_max_input_length: Tensor = None,
        freqs_cis: Tuple[Tensor, Tensor] = None,
    ) -> Tensor:
        mod_shift, mod_scale, mod_gate = (
            chunk(self.modulation(vec),3, dim=-1)
        )
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale, dtype=self.dtype)
        qkv, mlp = split(self.linear1(x_mod), [3 * self.hidden_size//self.mapping.tp_size, self.mlp_hidden_dim//self.mapping.tp_size], dim=-1)

        #q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        #B,L,_ = qkv.shape
        B = shape(qkv, 0)
        L = shape(qkv, 1)
        q, k, v = qkv.view(concat([B, L, 3, self.num_heads//self.mapping.tp_size, self.head_size])).permute([2,0,1,3,4]).unbind(dim=0)
        
        # Apply QK-Norm if needed.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if needed.
        constant_txt_len = constant(dims_array([txt_len]))
        if freqs_cis is not None:
            if self.reverse:
                slice_index = shape(q,1)-constant_txt_len
                img_q, txt_q = dynamic_split(q, [slice_index, constant_txt_len], dim=1)
                img_k, txt_k = dynamic_split(k, [slice_index, constant_txt_len], dim=1)
                #img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
                #img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
                img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
                img_q, img_k = img_qq, img_kk
                q = concat([img_q, txt_q], dim=1)
                k = concat([img_k, txt_k], dim=1)
            else:
                #txt_q, img_q = q[:, :txt_len, :, :], q[:, txt_len:, :, :]
                #txt_k, img_k = k[:, :txt_len, :, :], k[:, txt_len:, :, :]
                txt_q, img_q = dynamic_split(q, [constant_txt_len, shape(q,1)-constant_txt_len], dim=1)
                txt_k, img_k = dynamic_split(k, [constant_txt_len, shape(k,1)-constant_txt_len], dim=1)
                img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
                assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, \
                    f'img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}'
                img_q, img_k = img_qq, img_kk
                q = concat([txt_q, img_q], dim=1)
                k = concat([txt_k, img_k], dim=1)

        # Compute attention.
        # TODO(jarvizhang): Currently only support `torch` mode
        #attn = attention(q, k, v, mode=self.attn_mode, attn_mask=attn_mask)
        attn = self.core_attention(q, k, v, attention_mask=attn_mask, input_lengths=attn_input_length, max_input_length=attn_max_input_length)
        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(concat([attn, self.mlp_act(mlp)], 2))
        return x + apply_gate(output, gate=mod_gate)


