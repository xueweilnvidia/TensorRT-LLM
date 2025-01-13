/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/attentionMask.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include <cub/cub.cuh>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

// A stateful callback functor that maintains the running sum between consecutive scans.
struct BlockPrefixCallbackOp
{
    // Running prefix
    int mRunningTotal;

    // Constructor
    __device__ BlockPrefixCallbackOp(int runningTotal)
        : mRunningTotal(runningTotal)
    {
    }

    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ int operator()(int blockAggregate)
    {
        int oldPrefix = mRunningTotal;
        mRunningTotal += blockAggregate;
        return oldPrefix;
    }
};

// Given an array of sequence lengths, with batchSize elements, that kernel computes the exclusive
// prefix-sums of the sequence lengths. There are (batchSize+1) elements in seqOffsets.
//
// seqOffsets[ 0]        = 0
// seqOffsets[ii]        = seqLengths[0] + .. + seqLengths[ii-1],
// seqOffsets[batchSize] = seqLengths[0] + .. + seqLengths[batchSize-1]
//
// This kernel uses a single thread block of THREADS_PER_BLOCK threads.

// This kernel also computes the padding offsets: Given the index (idx) of a token in a ragged tensor,
// we need the index of the token in the corresponding tensor with padding. We compute an array
// of numTokens elements, called the paddingOffsets, such that the position in the padded tensor
// of the token "idx" in the ragged tensor is given by idx + paddingOffset[idx].
//
// That kernel uses a grid of batchSize blocks.

template <typename T, int THREADS_PER_BLOCK>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void computeSeqAndPaddingOffsets(BuildDecoderInfoParams<T> params)
{
    // Dynamic shared memory for storing seqOffsets.
    extern __shared__ int smem[];
    int* smemSeqQOffsets = (int*) (smem);

    // Fixed Q sequence lengths.
    bool const fixed_q_seqlen = params.seqQLengths == nullptr;

    // Whether to calculate cumulative KV sequence lengths.
    bool const calculate_kv_offsets = params.seqKVOffsets != nullptr;

    // Whether to calculate cumulative packed mask rows.
    bool const calculate_packed_mask_row_offsets = params.packedMaskRowOffsets != nullptr;

    // Compute the padding offsets for Encoder Inputs.
    bool const need_encoder_padding_offsets = (params.encoderPaddingOffsets != nullptr) && calculate_kv_offsets;
    [[maybe_unused]] int* smemEncoderSeqQOffsets;

    // The implementation of the parallel scan in the thread block (see CUB for details).
    using BlockScan = cub::BlockScan<int, THREADS_PER_BLOCK>;

    // Allocate storage in shared memory to do the scan.
    __shared__ typename BlockScan::TempStorage tempQStorage;
    [[maybe_unused]] __shared__ typename BlockScan::TempStorage tempMaskStorage;
    [[maybe_unused]] __shared__ typename BlockScan::TempStorage tempKVStorage;

    // This prefixOp operator keeps a running sum for when we need multiple iterations of the loop.
    BlockPrefixCallbackOp prefixQOp(0);
    BlockPrefixCallbackOp prefixMaskOp(0);
    BlockPrefixCallbackOp prefixKVOp(0);

    if (need_encoder_padding_offsets)
    {
        smemEncoderSeqQOffsets = (int*) (&smemSeqQOffsets[params.batchSize + 1]);
    }

    // Iterate over the sequences in the batch.
    //
    // The loop index does not depend on the thread index to make sure all the threads enter the
    // loop as we have __syncthreads in it (and we need all threads to participate to avoid
    // deadlocks).
    // Only the last block computes the full sequence offsets.
    bool const storeSeqOffsets = blockIdx.x == (params.batchSize - 1);
    int const batchSizeBound = blockIdx.x + 1;
    for (int batchOffset = 0; batchOffset <= batchSizeBound; batchOffset += THREADS_PER_BLOCK)
    {
        // The index of the batch.
        int batchIdx = batchOffset + threadIdx.x;

        // Threads that correspond to valid sequences read the length.
        int seqQLength = 0;
        [[maybe_unused]] int packedMaskRows = 0;
        [[maybe_unused]] int seqKVLength = 0;
        if (batchIdx < batchSizeBound)
        {
            seqQLength = fixed_q_seqlen ? params.maxQSeqLength : params.seqQLengths[batchIdx];
            // Need to pad mask rows to multiple of 128 for each sequence in the batch.
            packedMaskRows = calculate_packed_mask_row_offsets
                ? divUp(seqQLength, int(FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT)) * FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT
                : 0;
            seqKVLength = calculate_kv_offsets ? params.seqKVLengths[batchIdx] : 0;
        }

        // Do the prefix-scan (it calls syncthreads internally).
        int seqQOffset;
        [[maybe_unused]] int packedMaskRowOffset;
        [[maybe_unused]] int seqKVOffset;
        BlockScan(tempQStorage).ExclusiveSum(seqQLength, seqQOffset, prefixQOp);
        if (calculate_packed_mask_row_offsets)
        {
            BlockScan(tempMaskStorage).ExclusiveSum(packedMaskRows, packedMaskRowOffset, prefixMaskOp);
        }
        if (calculate_kv_offsets)
        {
            BlockScan(tempKVStorage).ExclusiveSum(seqKVLength, seqKVOffset, prefixKVOp);
        }

        // Store the result to smem.
        if (batchIdx <= batchSizeBound)
        {
            smemSeqQOffsets[batchIdx] = seqQOffset;
            if (need_encoder_padding_offsets)
            {
                smemEncoderSeqQOffsets[batchIdx] = seqKVOffset;
            }
        }

        // Store the result.
        if (batchIdx <= batchSizeBound && storeSeqOffsets)
        {
            params.seqQOffsets[batchIdx] = seqQOffset;
            if (calculate_packed_mask_row_offsets)
            {
                params.packedMaskRowOffsets[batchIdx] = packedMaskRowOffset;
            }
            if (calculate_kv_offsets)
            {
                params.seqKVOffsets[batchIdx] = seqKVOffset;
            }
        }

        // Make sure the shared memory can be reused for the next iteration of the loop.
        __syncthreads();
    }

    int batchIdx = blockIdx.x;

    // Compute the padding offsets.
    auto compute_padding_offset = [&](int* smem_offset, int maxSeqLength, int* paddingOffsets)
    {
        // Block x dimension is the batch dimension, while threads iterate all tokens in the sequence.
        int seqBegin = smem_offset[batchIdx];
        // The offset to the 1st element of the next sequence.
        int seqEnd = smem_offset[batchIdx + 1];
        // The length of the sequence.
        int seqLength = seqEnd - seqBegin;
        // The number of padded tokens in the previous sequences.
        int paddingOffset = batchIdx * maxSeqLength - seqBegin;

        // Iterate over the tokens to update the number of padded elements.
        for (int tokenIdx = threadIdx.x; tokenIdx < seqLength; tokenIdx += blockDim.x)
        {
            paddingOffsets[seqBegin + tokenIdx] = paddingOffset;
        }
    };

    if (params.paddingOffsets != nullptr)
    {
        compute_padding_offset(smemSeqQOffsets, params.maxQSeqLength, params.paddingOffsets);
    }

    if (need_encoder_padding_offsets)
    {
        compute_padding_offset(smemEncoderSeqQOffsets, params.maxEncoderQSeqLength, params.encoderPaddingOffsets);
    }

    // Each block generates the rotary embedding inv_freq tensor for the corresponding sequence.
    int zid = 2 * threadIdx.x;
    int halfRotaryEmbeddingDim = params.rotaryEmbeddingDim / 2;
    if (params.rotaryEmbeddingDim > 0 && zid < params.rotaryEmbeddingDim)
    {
        mmha::update_rotary_base_n_scale(params.rotaryEmbeddingBase, params.rotaryEmbeddingScale,
            params.rotaryScalingType, params.rotaryEmbeddingDim, params.rotaryEmbeddingMaxPositions,
            params.seqKVLengths[batchIdx]);
        // Recompute the rotary scales when it is dynamic scaling.
        if (params.rotaryScalingType == RotaryScalingType::kDYNAMIC || params.rotaryEmbeddingInvFreqCache == nullptr)
        {
            float const invFreq = params.rotaryEmbeddingScale
                / powf(params.rotaryEmbeddingBase, zid / (float) params.rotaryEmbeddingDim);
            params.rotaryEmbeddingInvFreq[batchIdx * halfRotaryEmbeddingDim + threadIdx.x] = invFreq;
        }
        else
        {
            // Otherwise, expand the inv freq cache to batch size.
            float const invFreqCache = params.rotaryEmbeddingInvFreqCache[threadIdx.x];
            params.rotaryEmbeddingInvFreq[batchIdx * halfRotaryEmbeddingDim + threadIdx.x] = invFreqCache;
        }
    }

    // Prepare values for fmha.
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        bool is_sage = true;
        // Reset fmha tile counter to 0 before launching fmha kernels.
        if (params.fmhaTileCounter)
        {
            params.fmhaTileCounter[0] = 0u;
        }
        // Take the quantization scales into consideration.
        if (params.fmhaBmm1Scale)
        {
            // The scale after fmha bmm1.
            if (is_sage){
                params.fmhaBmm1Scale[0] = params.fmhaHostBmm1Scale;
            }else{
                params.fmhaBmm1Scale[0] = params.dequantScaleQkv[0] * params.dequantScaleQkv[0] * params.fmhaHostBmm1Scale;
            }
            // The scale prepared for log2 optimization.
            constexpr float kLog2e = 1.4426950408889634074f;
            params.fmhaBmm1Scale[1] = params.fmhaBmm1Scale[0] * kLog2e;
        }
        if (params.fmhaBmm2Scale)
        {
            // The scale after fmha bmm2.
            if(is_sage){
                params.fmhaBmm2Scale[0] = 1.0f;
            }else{
                params.fmhaBmm2Scale[0] = params.quantScaleO[0] * params.dequantScaleQkv[0];
            }
        }
    }
}

template <typename T>
void invokeBuildDecoderInfo(BuildDecoderInfoParams<T> const& params, cudaStream_t stream)
{
    // Compute the sequence and padding offsets.
    int const THREADS_PER_BLOCK = 256;
    TLLM_CHECK_WITH_INFO(params.rotaryEmbeddingDim / 2 <= 256 && params.rotaryEmbeddingDim % 2 == 0,
        "Rotary embedding dim is assumed to be smaller than 512 and multiple of 2.");
    TLLM_CHECK_WITH_INFO(
        !(params.seqKVLengths == nullptr && params.rotaryEmbeddingDim > 0), "KV sequence lengths buffer is invalid.");
    bool const need_encoder_padding_offsets
        = (params.encoderPaddingOffsets != nullptr) && (params.seqKVOffsets != nullptr);
    const size_t smem_size
        = (need_encoder_padding_offsets ? (params.batchSize + 1) * 2 : (params.batchSize + 1)) * sizeof(int);
    computeSeqAndPaddingOffsets<T, THREADS_PER_BLOCK>
        <<<params.batchSize, THREADS_PER_BLOCK, smem_size, stream>>>(params);

    // Compute the attention mask, if needed.
    if (params.attentionMask != nullptr)
    {
        TLLM_CHECK_WITH_INFO(params.seqQLengths != nullptr, "Q sequence lengths buffer is invalid.");
        AttentionMaskParams<T> attentionMaskParams;
        memset((void*) &attentionMaskParams, 0, sizeof(attentionMaskParams));
        // Set parameters.
        attentionMaskParams.mask = params.attentionMask;
        // Nullptr indicates that the row dimension are not packed (i.e. paddings are not removed).
        attentionMaskParams.cuQSeqLens = nullptr;
        attentionMaskParams.actualQSeqLens = params.seqQLengths;
        attentionMaskParams.actualKvSeqLens = params.seqQLengths;
        attentionMaskParams.attentionMaskType = params.attentionMaskType;
        attentionMaskParams.blockSparseParams = params.blockSparseParams;
        attentionMaskParams.batchSize = params.batchSize;
        attentionMaskParams.maxQSeqLen = params.maxQSeqLength;
        attentionMaskParams.maxKvSeqLen = params.maxQSeqLength;
        attentionMaskParams.slidingWindowSize = params.attentionWindowSize;
        // Launch the kernel.
        invokeBuildAttentionMask(attentionMaskParams, stream);
    }
}

template void invokeBuildDecoderInfo(BuildDecoderInfoParams<float> const&, cudaStream_t);
template void invokeBuildDecoderInfo(BuildDecoderInfoParams<half> const&, cudaStream_t);
#ifdef ENABLE_BF16
template void invokeBuildDecoderInfo(BuildDecoderInfoParams<__nv_bfloat16> const&, cudaStream_t);
#endif
#ifdef ENABLE_FP8
template void invokeBuildDecoderInfo(BuildDecoderInfoParams<__nv_fp8_e4m3> const&, cudaStream_t);
#endif

__global__ void updatePaddingCountKernel(int* paddingPerSeq, int const* seqLengths, int maxQSeqLength, int batchSize)
{

    for (int ii = threadIdx.x; ii < batchSize; ii += blockDim.x)
    {
        paddingPerSeq[ii] = maxQSeqLength - seqLengths[ii];
    }
}


template void sage_quant<128, 64, 64, 256, __nv_bfloat16, __nv_fp8_e4m3, float>(
    // host var
    unsigned int batch_size, unsigned int head_num, unsigned int max_seq_len,
    bool smooth_k, bool is_padded,
    // device input
    const void* q, const void* k, const void* v,
    const int stride_q, const int stride_k, const int stride_v,
    const int* cu_seqlens_q, const int* cu_seqlens_kv,
    // sizeof(workspace) = batch_size * head_num * head_size * sizeof(TSmoothK)
    void* workspace,
    // device output
    void* quant_q, void* quant_k, void* quant_v,
    float* scales_q, float* scales_k, float* scales_v,
    cudaStream_t stream);


template <
    int HeadSize,
    int kThreadCount,
    int kTokensPerThreadBlock,
    typename T,
    typename TSmoothK
>
__global__ void k_mean_kernel(
    const bool is_padded,
    const int max_seq_len,
    const int head_num,
    const void* k,
    const int stride_k,
    const int* cu_seqlens_kv,
    void* k_mean
) {
    int batch_id = blockIdx.y / head_num;
    int head_id = blockIdx.y % head_num;
    int channel_id = blockIdx.x * kThreadCount + threadIdx.x;

    if (channel_id >= HeadSize)
        return;

    int seq_start = cu_seqlens_kv[batch_id];
    int seq_len = cu_seqlens_kv[batch_id + 1] - seq_start;
    if (is_padded)
        seq_start = batch_id * max_seq_len;
    int seq_end = seq_start + seq_len;

    seq_start += blockIdx.z * kTokensPerThreadBlock;
    if (seq_start >= seq_end)
        return;
    
    seq_end = min(seq_start + kTokensPerThreadBlock, seq_end);

    float channel_mean = 0.f;

    for (int seq_id = seq_start; seq_id < seq_end; seq_id++)
    {
        const T *input = reinterpret_cast<const T*>(k) + seq_id * stride_k + head_id * HeadSize + channel_id;
        channel_mean += static_cast<float>(*input);
        input += stride_k;
    }

    channel_mean /= static_cast<float>(seq_len);

    TSmoothK *output = reinterpret_cast<TSmoothK*>(k_mean) + batch_id * head_num * HeadSize + head_id * HeadSize + channel_id;
    
    atomicAdd(output, channel_mean);
}


template <
    int HeadSize,
    int BlockSizeQ,
    int BlockSizeK,
    int BlockSizeV,
    typename T,
    typename TQuant,
    typename TSmooth
>
__global__ void sage_quant_kernel(
    const void* q, const void* k, const void* v,
    const int stride_q, const int stride_k, const int stride_v,
    const int* cu_seqlens_q, const int* cu_seqlens_kv, const void* k_mean,
    int max_seq_len, bool smooth_k, bool is_padded,
    // output
    void* quant_q, void* quant_k, void* quant_v,
    float* scales_q, float* scales_k, float* scales_v) {
    
    int batch_id = blockIdx.z;
    int head_id = blockIdx.y / 3;
    int qkv_id = blockIdx.y % 3;
    int qblock_id = blockIdx.x;

    constexpr int kElementsAccess = sizeof(float4) / sizeof(T);
    constexpr int tbDimx = 128 / sizeof(float4);
    constexpr int tbDimy = 128 / tbDimx;
    constexpr int tbIterx = HeadSize / tbDimx / kElementsAccess;
    int col_id = threadIdx.x % tbDimx;
    int row_id = threadIdx.x / tbDimx;


    if (qkv_id == 0) {
        // Q

        int seq_start = cu_seqlens_q[batch_id];
        int seq_end = cu_seqlens_q[batch_id + 1];

        if (seq_start + qblock_id * BlockSizeQ >= seq_end)
            return;

        if (is_padded) {
            int seq_len = seq_end - seq_start;
            seq_start = batch_id * max_seq_len;
            seq_end = seq_start + seq_len;
        }

        int seq_id = seq_start + qblock_id * BlockSizeQ + row_id;
        constexpr int tbItery = BlockSizeQ / tbDimy;

        const T *input = reinterpret_cast<const T*>(q) + seq_id * stride_q + head_id * HeadSize + col_id * kElementsAccess;

        T local_input[tbItery * tbIterx * kElementsAccess];
        T local_amax = T(0);

        int seq_id_ = seq_id;
        for (int y_ = 0; y_ < tbItery; y_++) {

            T *local_input_ptr = local_input + y_ * tbIterx * kElementsAccess;
            const T *input_ptr = input + y_ * tbDimy * stride_q;

            if (seq_id_ < seq_end){
                for (int x_ = 0; x_ < tbIterx; x_++) {

                    *reinterpret_cast<float4 *>(local_input_ptr) = *reinterpret_cast<const float4 *>(input_ptr);
                
                    for (int i = 0; i < kElementsAccess; i++)
                    {
                        T value = __habs(local_input_ptr[i]);
                        if (value > local_amax)
                            local_amax = value;
                    }
                    
                    local_input_ptr += kElementsAccess;
                    input_ptr += tbDimx * kElementsAccess;
                }
            }
            else {
                break;
            }  

            seq_id_ += tbDimy;
        }

        /// CUB block level max
        using BlockReduce = cub::BlockReduce<T, 128>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ float s_block_amax;

        // Compute the block-wide max for thread0
        // cuda::maximum<>{}
        float aggregate = BlockReduce(temp_storage).Reduce(local_amax, cub::Max{});

        if (row_id == 0 && col_id == 0)
            s_block_amax = static_cast<float>(aggregate);

        __syncthreads();

        float block_scale = s_block_amax / 448 + 1e-4;

        int max_qblock_per_seq = (max_seq_len + BlockSizeQ - 1) / BlockSizeQ;
        float *scales_q_ptr = scales_q +
                              batch_id * (gridDim.y / 3) * max_qblock_per_seq +
                              head_id * max_qblock_per_seq +
                              qblock_id;
        *scales_q_ptr = block_scale;

        TQuant local_input_fp8[tbItery * tbIterx * kElementsAccess];

        for (int i = 0; i < tbItery * tbIterx * kElementsAccess; i++)
        {
            local_input_fp8[i] = static_cast<TQuant>(static_cast<float>(local_input[i]) / block_scale);
        }
        
        TQuant *output = reinterpret_cast<TQuant *>(quant_q) + seq_id * stride_q + head_id * HeadSize + col_id * kElementsAccess;

        for (int y_ = 0; y_ < tbItery; y_++) {

            TQuant *local_output_ptr = local_input_fp8 + y_ * tbIterx * kElementsAccess;
            TQuant *output_ptr = output + y_ * tbDimy * stride_q;

            if (seq_id >= seq_end)
                break;
            
            for (int x_ = 0; x_ < tbIterx; x_++) {

                *reinterpret_cast<float2 *>(output_ptr) = *reinterpret_cast<float2 *>(local_output_ptr);
                
                local_output_ptr += kElementsAccess;
                output_ptr += tbDimx * kElementsAccess;
            }
            
            seq_id += tbDimy;
        }
    }
    else if (qkv_id == 1) {
        // K

        int seq_start = cu_seqlens_kv[batch_id];
        int seq_end = cu_seqlens_kv[batch_id + 1];

        if (seq_start + qblock_id * BlockSizeK >= seq_end)
            return;

        if (is_padded) {
            int seq_len = seq_end - seq_start;
            seq_start = batch_id * max_seq_len;
            seq_end = seq_start + seq_len;
        }

        int seq_id = seq_start + qblock_id * BlockSizeK + row_id;
        constexpr int tbItery = BlockSizeK / tbDimy;

        const T *input = reinterpret_cast<const T*>(k) + seq_id * stride_k + head_id * HeadSize + col_id * kElementsAccess;

        TSmooth local_k_mean[tbIterx * kElementsAccess];

        if (smooth_k)
        {
            int head_num = gridDim.y / 3;
            const TSmooth *k_mean_ptr = reinterpret_cast<const TSmooth *>(k_mean) +
                                batch_id * head_num * HeadSize +
                                head_id * HeadSize +
                                col_id * kElementsAccess;
            for (int x_ = 0; x_ < tbIterx; x_++) {
                for (int i = 0; i < sizeof(TSmooth) / sizeof(T); i++) {
                    *(reinterpret_cast<float4 *>(local_k_mean + x_ * kElementsAccess) + i) = *(reinterpret_cast<const float4 *>(k_mean_ptr) + i);
                }

                k_mean_ptr += tbDimx * kElementsAccess;
            }
        }

        T local_input[tbItery * tbIterx * kElementsAccess];
        T local_amax = T(0);

        int seq_id_ = seq_id;
        for (int y_ = 0; y_ < tbItery; y_++) {

            T *local_input_ptr = local_input + y_ * tbIterx * kElementsAccess;
            const T *input_ptr = input + y_ * tbDimy * stride_k;

            if (seq_id_ < seq_end){
                for (int x_ = 0; x_ < tbIterx; x_++) {

                    *reinterpret_cast<float4 *>(local_input_ptr) = *reinterpret_cast<const float4 *>(input_ptr);
                
                    for (int i = 0; i < kElementsAccess; i++)
                    {
                        if (smooth_k)
                        {
                            local_input_ptr[i] -= local_k_mean[x_ * kElementsAccess + i];
                        }

                        T value = __habs(local_input_ptr[i]);
                        if (value > local_amax)
                            local_amax = value;
                    }
                    
                    local_input_ptr += kElementsAccess;
                    input_ptr += tbDimx * kElementsAccess;
                }
            }
            else {
                break;
            }  

            seq_id_ += tbDimy;
        }

        /// CUB block level max
        using BlockReduce = cub::BlockReduce<T, 128>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ float s_block_amax;

        // Compute the block-wide max for thread0
        // cuda::maximum<>{}
        float aggregate = BlockReduce(temp_storage).Reduce(local_amax, cub::Max{});

        if (row_id == 0 && col_id == 0)
            s_block_amax = static_cast<float>(aggregate);

        __syncthreads();

        float block_scale = s_block_amax / 448 + 1e-4;

        int max_qblock_per_seq = (max_seq_len + BlockSizeK - 1) / BlockSizeK;

        float *scales_ptr = scales_k +
                            batch_id * (gridDim.y / 3) * max_qblock_per_seq +
                            head_id * max_qblock_per_seq +
                            qblock_id;
        *scales_ptr = block_scale;

        TQuant *output = reinterpret_cast<TQuant *>(quant_k) + seq_id * stride_k + head_id * HeadSize + col_id * kElementsAccess;

        TQuant local_input_fp8[tbItery * tbIterx * kElementsAccess];

        for (int i = 0; i < tbItery * tbIterx * kElementsAccess; i++)
        {
            local_input_fp8[i] = static_cast<TQuant>(static_cast<float>(local_input[i]) / block_scale);
        }
        
        for (int y_ = 0; y_ < tbItery; y_++) {

            TQuant *local_output_ptr = local_input_fp8 + y_ * tbIterx * kElementsAccess;
            TQuant *output_ptr = output + y_ * tbDimy * stride_k;

            if (seq_id >= seq_end)
                break;
            
            for (int x_ = 0; x_ < tbIterx; x_++) {

                *reinterpret_cast<float2 *>(output_ptr) = *reinterpret_cast<float2 *>(local_output_ptr);
                
                local_output_ptr += kElementsAccess;
                output_ptr += tbDimx * kElementsAccess;
            }
            
            seq_id += tbDimy;
        }
    }
    else if (qkv_id == 2) {
        // V

        int seq_start = cu_seqlens_kv[batch_id];
        int seq_end = cu_seqlens_kv[batch_id + 1];

        if (seq_start + qblock_id * BlockSizeV >= seq_end)
            return;

        if (is_padded) {
            int seq_len = seq_end - seq_start;
            seq_start = batch_id * max_seq_len;
            seq_end = seq_start + seq_len;
        }

        int seq_id = seq_start + qblock_id * BlockSizeV + row_id;
        constexpr int tbItery = BlockSizeV / tbDimy;

        const T *input = reinterpret_cast<const T*>(v) + seq_id * stride_v + head_id * HeadSize + col_id * kElementsAccess;
        
        T local_input[tbItery * tbIterx * kElementsAccess];
        T local_amax = T(0);

        int seq_id_ = seq_id;
        for (int y_ = 0; y_ < tbItery; y_++) {

            T *local_input_ptr = local_input + y_ * tbIterx * kElementsAccess;
            const T *input_ptr = input + y_ * tbDimy * stride_v;

            if (seq_id_ < seq_end){
                for (int x_ = 0; x_ < tbIterx; x_++) {

                    *reinterpret_cast<float4 *>(local_input_ptr) = *reinterpret_cast<const float4 *>(input_ptr);
                
                    for (int i = 0; i < kElementsAccess; i++)
                    {
                        T value = __habs(local_input_ptr[i]);
                        if (value > local_amax)
                            local_amax = value;
                    }
                    
                    local_input_ptr += kElementsAccess;
                    input_ptr += tbDimx * kElementsAccess;
                }
            }
            else {
                break;
            }  

            seq_id_ += tbDimy;
        }

        /// CUB block level max
        using BlockReduce = cub::BlockReduce<T, 128>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ float s_block_amax;

        // Compute the block-wide max for thread0
        // cuda::maximum<>{}
        float aggregate = BlockReduce(temp_storage).Reduce(local_amax, cub::Max{});

        if (row_id == 0 && col_id == 0)
            s_block_amax = static_cast<float>(aggregate);

        __syncthreads();

        float block_scale = s_block_amax / 448 + 1e-4;

        int max_qblock_per_seq = (max_seq_len + BlockSizeV - 1) / BlockSizeV;

        float *scales_ptr = scales_v +
                            batch_id * (gridDim.y / 3) * max_qblock_per_seq +
                            head_id * max_qblock_per_seq +
                            qblock_id;
        *scales_ptr = block_scale;

        TQuant *output = reinterpret_cast<TQuant *>(quant_v) + seq_id * stride_v + head_id * HeadSize + col_id * kElementsAccess;

        TQuant local_input_fp8[tbItery * tbIterx * kElementsAccess];

        for (int i = 0; i < tbItery * tbIterx * kElementsAccess; i++)
        {
            local_input_fp8[i] = static_cast<TQuant>(static_cast<float>(local_input[i]) / block_scale);
        }
        
        for (int y_ = 0; y_ < tbItery; y_++) {

            TQuant *local_output_ptr = local_input_fp8 + y_ * tbIterx * kElementsAccess;
            TQuant *output_ptr = output + y_ * tbDimy * stride_v;

            if (seq_id >= seq_end)
                break;
            
            for (int x_ = 0; x_ < tbIterx; x_++) {

                *reinterpret_cast<float2 *>(output_ptr) = *reinterpret_cast<float2 *>(local_output_ptr);
                
                local_output_ptr += kElementsAccess;
                output_ptr += tbDimx * kElementsAccess;
            }
            
            seq_id += tbDimy;
        }
    }
}



template <
    int HeadSize,
    int BlockSizeQ,
    int BlockSizeK,
    int BlockSizeV,
    typename T,
    typename TQuant,
    typename TSmoothK
>
void sage_quant(
    // host var
    unsigned int batch_size, unsigned int head_num, unsigned int max_seq_len,
    bool smooth_k, bool is_padded,
    // device input
    const void* q, const void* k, const void* v,
    const int stride_q, const int stride_k, const int stride_v,
    const int* cu_seqlens_q, const int* cu_seqlens_kv,
    // sizeof(workspace) = batch_size * head_num * head_size * sizeof(TSmoothK)
    void* workspace,
    // device output
    void* quant_q, void* quant_k, void* quant_v,
    float* scales_q, float* scales_k, float* scales_v,
    cudaStream_t stream)
{
    // if (BlockSizeQ % BlockSizeKV != 0) {
    //     printf("Failed, BlockSizeQ should be divisible by BlockSizeKV. %s:%d\n", __FILE__, __LINE__);                      \
    //     exit(EXIT_FAILURE);
    // }

    void* k_mean = workspace;

    if (smooth_k)
    {

        const int tokens_per_block = 1024;
        const int block = 128;
        dim3 grid(
            (HeadSize + block - 1) / block,
            batch_size * head_num,
            (max_seq_len + tokens_per_block - 1) / tokens_per_block
        );

        cudaMemsetAsync(k_mean, 0, batch_size * head_num * HeadSize * sizeof(TSmoothK), stream);
        k_mean_kernel<HeadSize, block, tokens_per_block, T, TSmoothK><<<grid, block, 0, stream>>>(
            is_padded,
            max_seq_len,
            head_num,
            k,
            stride_k,
            cu_seqlens_kv,
            k_mean);
    }

    constexpr int BlockSize_ = (BlockSizeQ > BlockSizeK) ? BlockSizeK : BlockSizeQ;
    constexpr int BlockSize = (BlockSizeV > BlockSize_) ? BlockSize_ : BlockSizeV;

    dim3 grid(
        (max_seq_len + BlockSize - 1) / BlockSize,
        head_num * 3,
        batch_size
    );

    sage_quant_kernel<HeadSize, BlockSizeQ, BlockSizeK, BlockSizeV,
                    T, TQuant, TSmoothK><<<grid, 128, 0, stream>>>(
        q, k, v,
        stride_q, stride_k, stride_v,
        cu_seqlens_q, cu_seqlens_kv, k_mean,
        max_seq_len, smooth_k, is_padded,
        quant_q, quant_k, quant_v,
        scales_q, scales_k, scales_v);
}

} // namespace kernels
} // namespace tensorrt_llm
