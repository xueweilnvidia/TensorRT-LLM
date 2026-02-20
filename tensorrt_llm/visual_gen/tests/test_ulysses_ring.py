import math
import torch
import torch.distributed as dist
import torch.nn.functional as F

from visual_gen.configs.op_manager import AttentionOpManager
from visual_gen.configs.parallel import get_dit_parallel_config
from visual_gen.configs.parallel import DiTParallelConfig
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.layers.attention import ditAttnProcessor
from visual_gen.layers.utils import ulysses_a2a_in, ulysses_a2a_out
from visual_gen.utils import get_logger

logger = get_logger(__name__)

import nvtx


def sample_tensors(num_heads, seq_len, head_dim, world_size):
    """Create sample tensors for attention testing."""

    shape = (num_heads, seq_len, head_dim)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    # Prepare inputs
    q = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    local_q = q.chunk(world_size, dim=1)[rank]
    local_k = k.chunk(world_size, dim=1)[rank]
    local_v = v.chunk(world_size, dim=1)[rank]
    return q, k, v, local_q, local_k, local_v





def test_ulysses_communication(num_heads, seq_len, head_dim, world_size, tensor_layout):
    """Test Ulysses communication functionality."""
    
    rank = dist.get_rank()
    PipelineConfig.reset()
    q, k, v, local_q, local_k, local_v = sample_tensors(num_heads, seq_len, head_dim, world_size)

    dit_config = DiTParallelConfig()
    dit_config.set_config(
        tp_size=1,
        cfg_size=1,
        ulysses_size=world_size,
        ring_size=1,
    )

    ulysses_group = get_dit_parallel_config().ulysses_group()

    if tensor_layout == "NHD":
        local_q = local_q.permute(1, 0, 2).contiguous()
        local_k = local_k.permute(1, 0, 2).contiguous()
        local_v = local_v.permute(1, 0, 2).contiguous()

        q = q.permute(1, 0, 2).contiguous()
        k = k.permute(1, 0, 2).contiguous()
        v = v.permute(1, 0, 2).contiguous()

    with nvtx.annotate(f"ulysses_a2a_in"):
        q_ulysses, k_ulysses, v_ulysses, _ = ulysses_a2a_in(
            local_q,
            local_k,
            local_v,
            None,
            tensor_layout,
            ulysses_size=world_size,
            ulysses_rank=rank,
            ulysses_group=ulysses_group,
            int8_all2all=False,
            fuse_qkv=True,
        )

    if tensor_layout == "NHD":
        chunk_dim = 1
    elif tensor_layout == "HND":
        chunk_dim = 0
    else:
        raise ValueError(f"Invalid tensor layout: {tensor_layout}")

    q_ref = q.chunk(world_size, dim=chunk_dim)[rank]
    k_ref = k.chunk(world_size, dim=chunk_dim)[rank]
    v_ref = v.chunk(world_size, dim=chunk_dim)[rank]
    
    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_similarity = cos_sim(q_ulysses.reshape(-1).to(torch.float32), q_ref.reshape(-1).to(torch.float32))
    print("cos_similarity q: ", cos_similarity)
    cos_similarity = cos_sim(k_ulysses.reshape(-1).to(torch.float32), k_ref.reshape(-1).to(torch.float32))
    print("cos_similarity k: ", cos_similarity)
    cos_similarity = cos_sim(v_ulysses.reshape(-1).to(torch.float32), v_ref.reshape(-1).to(torch.float32))
    print("cos_similarity v: ", cos_similarity)

def split_varlen_tensor(tensor, seq_lens, num_chunks, rank, seq_len_padded_cur_rank, tensor_layout="HND"):
    if tensor_layout == "NHD":
        chunk_dim = 0
    elif tensor_layout == "HND":
        chunk_dim = 1
    else:
        raise ValueError(f"Invalid tensor layout: {tensor_layout}")

    """Split a concatenated variable-length tensor into equal chunks across ranks.

    Given a tensor of shape [total_seq_len, head_num, head_dim] where total_seq_len
    is the sum of multiple sub-sequences, split each sub-sequence into `num_chunks`
    parts and return the `rank`-th chunk concatenated together.

    For each sub-sequence, the first (num_chunks - 1) ranks each get
    ceil(seq_len / num_chunks) elements, and the last rank gets whatever
    remains.

    Args:
        tensor: Tensor of shape [total_seq_len, head_num, head_dim].
        seq_lens: List or 1-D tensor of individual sequence lengths that sum to
                  total_seq_len, e.g. [1021, 1024, 1027].
        num_chunks: Number of chunks to split into (typically world_size, e.g. 4).
        rank: Which chunk to return (0-indexed).

    Returns:
        A tensor of shape [chunk_seq_len, head_num, head_dim] where chunk_seq_len
        is the sum of the rank-th chunk of every sub-sequence.
    """
    if isinstance(seq_lens, torch.Tensor):
        seq_lens = seq_lens.tolist()

    chunks = []
    offset = 0
    for seq_len in seq_lens:
        seq_len = int(seq_len)
        # First (num_chunks - 1) ranks get ceil(seq_len / num_chunks),
        # last rank gets whatever is left.
        base = math.ceil(seq_len / num_chunks)
        if rank < num_chunks - 1:
            chunk_len = base
            start = offset + base * rank
        else:
            # Last rank gets the remainder
            start = offset + base * (num_chunks - 1)
            chunk_len = seq_len - base * (num_chunks - 1)

        end = start + chunk_len
        chunks.append(tensor.narrow(chunk_dim, start, chunk_len))
        offset += seq_len

    res = torch.cat(chunks, dim=chunk_dim)

    if res.shape[chunk_dim] < seq_len_padded_cur_rank:
        pad_len = seq_len_padded_cur_rank - res.shape[chunk_dim]
        pad_shape = list(res.shape)
        pad_shape[chunk_dim] = pad_len
        res = torch.cat([res, torch.zeros(pad_shape, device=res.device, dtype=res.dtype)], dim=chunk_dim)
    
    return res


def varlen_cp_config(actual_seq_lens_q, actual_seq_lens_kv, world_size):

    # world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank}")
    # print(f"rank: {rank}, actual_seq_lens_q: {actual_seq_lens_q}, actual_seq_lens_kv: {actual_seq_lens_kv}")

    padded_seq_lens_q = torch.ceil(actual_seq_lens_q / world_size) * world_size
    padded_seq_lens_kv = torch.ceil(actual_seq_lens_kv / world_size) * world_size

    # print(f"rank: {rank}, padded_seq_lens_q: {padded_seq_lens_q}, padded_seq_lens_kv: {padded_seq_lens_kv}")

    padded_seq_len_q_cur_rank = torch.floor(padded_seq_lens_q / world_size).to(torch.int32)
    padded_seq_len_kv_cur_rank = torch.floor(padded_seq_lens_kv / world_size).to(torch.int32)

    max_seq_len_q = padded_seq_len_q_cur_rank.max()
    max_seq_len_kv = padded_seq_len_kv_cur_rank.max()

    # print(f"max_seq_len_q: {max_seq_len_q}, max_seq_len_kv: {max_seq_len_kv}")

    cu_seqlens_q_all_ranks = []
    cu_seqlens_kv_all_ranks = []

    for i in range(world_size):
        
        if i == world_size - 1:
            seq_len_q_cur_rank = padded_seq_len_q_cur_rank - (padded_seq_lens_q - actual_seq_lens_q)
            seq_len_kv_cur_rank = padded_seq_len_kv_cur_rank - (padded_seq_lens_kv - actual_seq_lens_kv)
        else:
            seq_len_q_cur_rank = padded_seq_len_q_cur_rank
            seq_len_kv_cur_rank = padded_seq_len_kv_cur_rank

        cu_seqlens_q = [0]
        for seq_len in seq_len_q_cur_rank:
            cu_seqlens_q.append(cu_seqlens_q[-1] + seq_len)
        cu_seqlens_q_all_ranks.append(torch.tensor(cu_seqlens_q).to(device).to(torch.int32))

        cu_seqlens_kv = [0]
        for seq_len in seq_len_kv_cur_rank:
            cu_seqlens_kv.append(cu_seqlens_kv[-1] + seq_len)
        cu_seqlens_kv_all_ranks.append(torch.tensor(cu_seqlens_kv).to(device).to(torch.int32))
        
    # print(f"cu_seqlens_q_all_ranks: {cu_seqlens_q_all_ranks}, cu_seqlens_kv_all_ranks: {cu_seqlens_kv_all_ranks}")

    return torch.stack(cu_seqlens_q_all_ranks), torch.stack(cu_seqlens_kv_all_ranks), max_seq_len_q, max_seq_len_kv


def sample_varlen_tensors(num_heads, head_dim, world_size, seq_len_list):
   
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    total_seq_len = sum(seq_len_list)
    seq_len_padded = torch.ceil(seq_len_list / world_size).to(torch.int32) * world_size
    total_seq_len_padded = sum(seq_len_padded)
    seq_len_padded_cur_rank = torch.ceil(total_seq_len_padded / world_size).to(torch.int32)


    print(f"total_seq_len_padded: {total_seq_len_padded}, seq_len_padded_cur_rank: {seq_len_padded_cur_rank}")

    shape = (num_heads, total_seq_len, head_dim)

    q = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    local_q = split_varlen_tensor(q, seq_len_list, world_size, rank, seq_len_padded_cur_rank)
    local_k = split_varlen_tensor(k, seq_len_list, world_size, rank, seq_len_padded_cur_rank)
    local_v = split_varlen_tensor(v, seq_len_list, world_size, rank, seq_len_padded_cur_rank)
    
    return q, k, v, local_q, local_k, local_v


def test_uneven_varlen_attn_parallel(
    num_heads, seq_len_list, head_dim, world_size, ulysses_size, ring_size, attn_type
):
    """Test uneven parallel attention functionality."""

    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    

    total_seq_len = sum(seq_len_list)
    seq_len_padded = torch.ceil(seq_len_list / world_size).to(torch.int32) * world_size
    total_seq_len_padded = sum(seq_len_padded)
    seq_len_padded_cur_rank = torch.ceil(total_seq_len_padded / world_size).to(torch.int32)

    full_cu_seqlens = [0]
    for seq_len in seq_len_list:
        full_cu_seqlens.append(full_cu_seqlens[-1] + seq_len)

    PipelineConfig.reset()

    cu_seqlens_q_all_ranks, cu_seqlens_kv_all_ranks, max_seq_len_q, max_seq_len_kv = varlen_cp_config(seq_len_list, seq_len_list, world_size)
    query, key, value, local_query, local_key, local_value = sample_varlen_tensors(num_heads, head_dim, world_size, seq_len_list)

    # print(f"query.shape: {query.shape}, key.shape: {key.shape}, value.shape: {value.shape}")
    # print(f"local_query.shape: {local_query.shape}, local_key.shape: {local_key.shape}, local_value.shape: {local_value.shape}")
    print(f"cu_seqlens_q_all_ranks: {cu_seqlens_q_all_ranks}, cu_seqlens_kv_all_ranks: {cu_seqlens_kv_all_ranks}")

    dit_config = DiTParallelConfig()
    dit_config.set_config(
        tp_size=1,
        cfg_size=1,
        ulysses_size=ulysses_size,
        ring_size=ring_size,
    )
    
    # PipelineConfig.set_uneven_cp_config(total_seq_len, seq_len_padded, seq_len_cur_rank, dit_config)
    PipelineConfig.set_varlen_uneven_cp_config(cu_seqlens_q_all_ranks, cu_seqlens_kv_all_ranks, max_seq_len_q, max_seq_len_kv, dit_config)
    attn = ditAttnProcessor()
    AttentionOpManager.set_attn_config(attn_type=attn_type)
    local_output = attn.visual_gen_attn(local_query, local_key, local_value, tensor_layout="HND")

    local_ref_output_list = []
    for i in range(len(seq_len_list)):
        q_tmp = query[:, full_cu_seqlens[i]:full_cu_seqlens[i+1], :]
        k_tmp = key[:, full_cu_seqlens[i]:full_cu_seqlens[i+1], :]
        v_tmp = value[:, full_cu_seqlens[i]:full_cu_seqlens[i+1], :]
        tmp_output = F.scaled_dot_product_attention(q_tmp.unsqueeze(0), k_tmp.unsqueeze(0), v_tmp.unsqueeze(0), is_causal=False)
        local_ref_output_list.append(tmp_output)

    ref_output = torch.cat(local_ref_output_list, dim=2).squeeze(0)
    local_ref_output = split_varlen_tensor(ref_output, seq_len_list, world_size, rank, seq_len_padded_cur_rank)
    
    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_similarity = cos_sim(local_output.reshape(-1).to(torch.float32), local_ref_output.reshape(-1).to(torch.float32))
    print("cos_similarity total: ", cos_similarity)
    if cos_similarity < 0.99:
        print("local_output: ", local_output)
        print("local_ref_output: ", local_ref_output)
        raise RuntimeError("Accuracy test failed")



if __name__ == "__main__":
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = torch.distributed.get_rank()

    # test_ulysses_communication(24, 8 * 1024, 128, world_size, "HND")
    # test_ulysses_communication(24, 8 * 1024, 128, world_size, "HND")
    # test_ulysses_communication(24, 8 * 1024, 128, world_size, "NHD")

    # cu_seqlens_q_all_ranks, cu_seqlens_kv_all_ranks, max_seq_len_q, max_seq_len_kv = varlen_cp_config(torch.tensor([1021, 1024, 1027],dtype=torch.int32), 
    #                                                 torch.tensor([1021, 1024, 1027], dtype=torch.int32), 
    #                                                 world_size)

    # PipelineConfig.reset()
    # dit_config = DiTParallelConfig()
    # dit_config.set_config(
    #     tp_size=1,
    #     cfg_size=1,
    #     ulysses_size=1,
    #     ring_size=world_size,
    # )
    # PipelineConfig.set_varlen_uneven_cp_config(cu_seqlens_q_all_ranks, cu_seqlens_kv_all_ranks, max_seq_len_q, max_seq_len_kv, dit_config)


    test_uneven_varlen_attn_parallel(
            num_heads=24,
            seq_len_list=torch.tensor([1021, 1024, 1027, 750, 826], dtype=torch.int32),
            head_dim=128,
            world_size=world_size,
            ulysses_size=1,
            ring_size=world_size,
            attn_type="flash-attn3",
        )

    dist.destroy_process_group() 
