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
        // Reset fmha tile counter to 0 before launching fmha kernels.
        if (params.fmhaTileCounter)
        {
            params.fmhaTileCounter[0] = 0u;
        }
        // Take the quantization scales into consideration.
        if (params.fmhaBmm1Scale)
        {
            // The scale after fmha bmm1.
            params.fmhaBmm1Scale[0] = params.dequantScaleQkv[0] * params.dequantScaleQkv[0] * params.fmhaHostBmm1Scale;
            // The scale prepared for log2 optimization.
            constexpr float kLog2e = 1.4426950408889634074f;
            params.fmhaBmm1Scale[1] = params.fmhaBmm1Scale[0] * kLog2e;
        }
        if (params.fmhaBmm2Scale)
        {
            // The scale after fmha bmm2.
            params.fmhaBmm2Scale[0] = params.quantScaleO[0] * params.dequantScaleQkv[0];
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



template void sage_quant<128, 128, 64, __nv_bfloat16, __nv_fp8_e4m3>(
    // host var
    unsigned int batch_size, unsigned int head_num, unsigned int head_size, unsigned int max_seq_len,
    // device var
    const void* q, const void* k, const void* v,
    int stride_q, int stride_k, int stride_v,
    const int* cu_seqlens_q, const int* cu_seqlens_kv,
    // int block_size_q, int block_size_k, int block_size_v,
    // output
    void* quant_q, void* quant_k, void* quant_v,
    float* scales_q, float* scales_k, float* scales_v);

template <
    int HeadSize,
    int BlockSizeQ,
    int BlockSizeKV,
    typename T,
    typename T_quant
>
__global__ void sage_quant_kernel(
    const void* q, const void* k, const void* v,
    int stride_q, int stride_k, int stride_v,
    const int* cu_seqlens_q, const int* cu_seqlens_kv,
    int max_seq_len,
    // int block_size_q, int block_size_k, int block_size_v,
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
                for (int i = 0; i < tbIterx * kElementsAccess; i++) {
                    local_input_ptr[i] = T(0);
                }
                local_input_ptr += tbIterx * kElementsAccess;
            }  

            seq_id_ += tbDimy;
        }

        /// CUB block level max
        using BlockReduce = cub::BlockReduce<T, 128>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ float s_block_amax;

        // Compute the block-wide max for thread0
        // cuda::maximum<>{}
        int aggregate = BlockReduce(temp_storage).Reduce(local_amax, cub::Max{});

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

        T_quant local_input_fp8[tbItery * tbIterx * kElementsAccess];

        for (int i = 0; i < tbItery * tbIterx * kElementsAccess; i++)
        {
            local_input_fp8[i] = static_cast<T_quant>(static_cast<float>(local_input[i]) / block_scale);
        }
        
        T_quant *output = reinterpret_cast<T_quant *>(quant_q) + seq_id * stride_q + head_id * HeadSize + col_id * kElementsAccess;

        for (int y_ = 0; y_ < tbItery; y_++) {

            T_quant *local_output_ptr = local_input_fp8 + y_ * tbIterx * kElementsAccess;
            T_quant *output_ptr = output + y_ * tbDimy * stride_q;

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
    else {
        // K(qkv_id == 1) & V(qkv_id == 2)

        int seq_start = cu_seqlens_kv[batch_id];
        int seq_end = cu_seqlens_kv[batch_id + 1];

        if (seq_start + qblock_id * BlockSizeKV >= seq_end)
            return;

        int seq_id = seq_start + qblock_id * BlockSizeKV + row_id;
        constexpr int tbItery = BlockSizeKV / tbDimy;

        const T *input;
        int stride;
        
        if (qkv_id == 1) {
            input = reinterpret_cast<const T*>(k);
            stride = stride_k;
        }
        else {
            input = reinterpret_cast<const T*>(v);
            stride = stride_v;
        }

        input += seq_id * stride + head_id * HeadSize + col_id * kElementsAccess;

        T local_input[tbItery * tbIterx * kElementsAccess];
        T local_amax = T(0);

        int seq_id_ = seq_id;
        for (int y_ = 0; y_ < tbItery; y_++) {

            T *local_input_ptr = local_input + y_ * tbIterx * kElementsAccess;
            const T *input_ptr = input + y_ * tbDimy * stride;

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
                for (int i = 0; i < tbIterx * kElementsAccess; i++) {
                    local_input_ptr[i] = T(0);
                }
                local_input_ptr += tbIterx * kElementsAccess;
            }  

            seq_id_ += tbDimy;
        }

        /// CUB block level max
        using BlockReduce = cub::BlockReduce<T, 128>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ float s_block_amax;

        // Compute the block-wide max for thread0
        // cuda::maximum<>{}
        int aggregate = BlockReduce(temp_storage).Reduce(local_amax, cub::Max{});

        if (row_id == 0 && col_id == 0)
            s_block_amax = static_cast<float>(aggregate);

        __syncthreads();

        float block_scale = s_block_amax / 448 + 1e-6;

        int max_qblock_per_seq = (max_seq_len + BlockSizeKV - 1) / BlockSizeKV;
        float *scales_ptr;
        T_quant *output;
        
        if (qkv_id == 1) {
            scales_ptr = scales_k;
            output = reinterpret_cast<T_quant *>(quant_k);
        }
        else {
            scales_ptr = scales_v;
            output = reinterpret_cast<T_quant *>(quant_v);
        }

        scales_ptr += batch_id * (gridDim.y / 3) * max_qblock_per_seq +
                      head_id * max_qblock_per_seq +
                      qblock_id;
        *scales_ptr = block_scale;

        T_quant local_input_fp8[tbItery * tbIterx * kElementsAccess];

        for (int i = 0; i < tbItery * tbIterx * kElementsAccess; i++)
        {
            local_input_fp8[i] = static_cast<T_quant>(static_cast<float>(local_input[i]) / block_scale);
        }
        
        output += seq_id * stride + head_id * HeadSize + col_id * kElementsAccess;

        for (int y_ = 0; y_ < tbItery; y_++) {

            T_quant *local_output_ptr = local_input_fp8 + y_ * tbIterx * kElementsAccess;
            T_quant *output_ptr = output + y_ * tbDimy * stride;

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
    int BlockSizeKV,
    typename T,
    typename T_quant
>
void sage_quant(
    // host var
    unsigned int batch_size, unsigned int head_num, unsigned int head_size, unsigned int max_seq_len,
    // device var
    const void* q, const void* k, const void* v,
    int stride_q, int stride_k, int stride_v,
    const int* cu_seqlens_q, const int* cu_seqlens_kv,
    // int block_size_q, int block_size_k, int block_size_v,
    // output
    void* quant_q, void* quant_k, void* quant_v,
    float* scales_q, float* scales_k, float* scales_v)
{
    // if (BlockSizeQ % BlockSizeKV != 0) {
    //     printf("Failed, BlockSizeQ should be divisible by BlockSizeKV. %s:%d\n", __FILE__, __LINE__);                      \
    //     exit(EXIT_FAILURE);
    // }

    constexpr int BlockSize = (BlockSizeQ > BlockSizeKV) ? BlockSizeKV : BlockSizeQ;

    dim3 grid(
        (max_seq_len + BlockSize - 1) / BlockSize,
        head_num * 3,
        batch_size
    );

    sage_quant_kernel<HeadSize, BlockSizeQ, BlockSizeKV, T, T_quant><<<grid, 128>>>(
        q, k, v,
        stride_q, stride_k, stride_v,
        cu_seqlens_q, cu_seqlens_kv,
        max_seq_len,
        // block_size_q, block_size_k, block_size_v,
        quant_q, quant_k, quant_v,
        scales_q, scales_k, scales_v);
    
    // cudaDeviceSynchronize();
    // fflush(stdout);
}





} // namespace kernels
} // namespace tensorrt_llm
