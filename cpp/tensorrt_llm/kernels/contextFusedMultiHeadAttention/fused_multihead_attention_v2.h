/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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

#pragma once
#include "cubin/fmha_cubin.h"
#include "cuda_runtime_api.h"
#include "fused_multihead_attention_common.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tmaDescriptor.h"
#include <algorithm>
#include <assert.h>
#include <memory>
#include <mutex>
#include <set>
#include <stdint.h>
#include <unordered_map>
#include <vector>
#include <string>

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// Base Class

template <typename TKernelMeta, typename TKernelParam>
class TFusedMultiHeadAttentionXMMAKernel
{
public:
    using KernelMeta = TKernelMeta;
    using KernelParam = TKernelParam;

    inline uint64_t hashID(unsigned int s, unsigned int d) const
    {
        return (uint64_t) s << 32 | d;
    }

    virtual uint64_t hashID(KernelMeta const& kernelMeta) const
    {
        return hashID(kernelMeta.mS, kernelMeta.mD);
    }

    TFusedMultiHeadAttentionXMMAKernel(
        TKernelMeta const* pMetaStart, unsigned int nMetaCount, Data_type type, unsigned int sm)
        : mDriver(tensorrt_llm::common::CUDADriverWrapper::getInstance())
        , mDataType(type)
        , mKernelMeta(pMetaStart)
        , mKernelMetaCount(nMetaCount)
        , mSM(sm)
    {
    }

    void loadXMMAKernels()
    {
        if (!mFunctions.empty())
        {
            return;
        }

        for (unsigned int i = 0; i < mKernelMetaCount; ++i)
        {
            auto const& kernelMeta = mKernelMeta[i];
            if (kernelMeta.mSM == mSM && kernelMeta.mDataType == mDataType)
            {
                CUmodule hmod{0};
                auto findModuleIter = mModules.find(kernelMeta.mCubin);
                if (findModuleIter != mModules.end())
                {
                    hmod = findModuleIter->second;
                }
                else
                {
                    cuErrCheck(mDriver->cuModuleLoadData(&hmod, kernelMeta.mCubin), mDriver);
                    mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
                }

                FusedMultiHeadAttentionKernelInfo funcInfo;
                funcInfo.mMetaInfoIndex = i;
                cuErrCheck(
                    mDriver->cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName), mDriver);
                if (kernelMeta.mSharedMemBytes >= 48 * 1024)
                {
                    cuErrCheck(mDriver->cuFuncSetAttribute(funcInfo.mDeviceFunction,
                                   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes),
                        mDriver);
                }
                std::string name = kernelMeta.mFuncName;
                std::string::size_type index = name.find("sage");
                if(index != std::string::npos){
                    std::cout<<name<<std::endl;
                    mFunctions.insert(std::make_pair(0, funcInfo));
                }else{
                    mFunctions.insert(std::make_pair(hashID(kernelMeta), funcInfo));
                }
                int s = static_cast<int>(kernelMeta.mS);
                if (mValidSequences.find(s) == mValidSequences.end())
                    mValidSequences.insert(s);
            }
        }
    }

    bool isValid(int s) const
    {
        return (mValidSequences.find(s) != mValidSequences.end());
    }

    virtual void run(TKernelParam& params, Launch_params& launch_params, cudaStream_t stream) const
    {
        auto const findIter = mFunctions.find(hashID(params.s, params.d));

        auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;

        void* kernelParams[] = {&params, nullptr};
        cuErrCheck(mDriver->cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                       kernelMeta.mSharedMemBytes, stream, kernelParams, nullptr),
            mDriver);
    }

    virtual bool checkIfKernelExist(MHARunnerFixedParams params) const = 0;

    virtual void getStepSize(uint32_t& out_step_q, uint32_t& out_step_kv, TKernelParam const& params,
        Launch_params const& launch_params) const
        = 0;

    virtual ~TFusedMultiHeadAttentionXMMAKernel() = default;

protected:
    struct FusedMultiHeadAttentionKernelInfo;

    struct KernelExistPredicate
    {
        KernelExistPredicate(uint64_t id)
            : mId(id)
        {
        }

        bool operator()(std::pair<uint64_t, FusedMultiHeadAttentionKernelInfo> const& v) const
        {
            return (v.first & mId) == mId;
        }

    private:
        uint64_t mId;
    };

protected:
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;

    Data_type mDataType;
    TKernelMeta const* mKernelMeta;
    unsigned int mKernelMetaCount;
    unsigned int mSM;
    std::unordered_map<unsigned char const*, CUmodule> mModules;

    struct FusedMultiHeadAttentionKernelInfo
    {
        unsigned int mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };

    std::unordered_map<uint64_t, FusedMultiHeadAttentionKernelInfo> mFunctions;
    std::set<int> mValidSequences;
};

template <typename TFusedMHAKernelList>
class TFusedMHAKernelFactory
{
public:
    TFusedMHAKernelList const* getXMMAKernels(const typename TFusedMHAKernelList::KernelMeta* pKernelList,
        unsigned int nbKernels, Data_type type, unsigned int sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        auto const id = hashID(type, sm);
        auto const findIter = mKernels.find(id);
        if (findIter == mKernels.end())
        {
            TFusedMHAKernelList* newKernel = new TFusedMHAKernelList{pKernelList, nbKernels, type, sm};
            newKernel->loadXMMAKernels();
            mKernels.insert(std::make_pair(id, std::unique_ptr<TFusedMHAKernelList>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static TFusedMHAKernelFactory<TFusedMHAKernelList>& Get()
    {
        int device_id;
        cudaGetDevice(&device_id);
        static std::unique_ptr<TFusedMHAKernelFactory<TFusedMHAKernelList>> s_factory[32] = {nullptr};
        if (s_factory[device_id] == nullptr)
        {
            assert(device_id <= 32);
            s_factory[device_id] = std::make_unique<TFusedMHAKernelFactory<TFusedMHAKernelList>>(
                TFusedMHAKernelFactory<TFusedMHAKernelList>());
        }

        return *(s_factory[device_id]);
    }

private:
    TFusedMHAKernelFactory() = default;

    inline uint64_t hashID(Data_type type, unsigned int sm) const
    {
        return (uint64_t) type << 32 | sm;
    }

    std::unordered_map<uint64_t, const std::unique_ptr<TFusedMHAKernelList>> mKernels;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

class FusedMultiHeadAttentionXMMAKernelV2
    : public TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoV2,
          Fused_multihead_attention_params_v2>
{
public:
    FusedMultiHeadAttentionXMMAKernelV2(FusedMultiHeadAttentionKernelMetaInfoV2 const* pMetaStart,
        unsigned int nMetaCount, Data_type type, unsigned int sm)
        : TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoV2,
            Fused_multihead_attention_params_v2>(pMetaStart, nMetaCount, type, sm)
    {
    }

    inline uint64_t hashID(unsigned int s, unsigned int d, bool interleaved, bool unroll, bool force_fp32_acc,
        bool flash_attention, bool warp_specialization, bool is_alibi_supported, int attention_mask_type,
        int input_layout, bool tiled, bool has_qk_tanh_scale) const
    {
        s = flash_attention ? 0 : s;
        // D <= 2048
        return (uint64_t) s << 32 | d << 16 | (attention_mask_type << 10) | (input_layout << 8)
            | (has_qk_tanh_scale ? 128ull : 0ull) | (is_alibi_supported ? 64ull : 0ull)
            | (warp_specialization ? 32ull : 0ull) | (tiled ? 16ull : 0ull) | (force_fp32_acc ? 8ull : 0ull)
            | (flash_attention ? 4ull : 0ull) | (interleaved ? 2ull : 0ull) | (unroll ? 1ull : 0ull);
    }

    uint64_t hashID(KernelMeta const& kernelMeta) const override
    {

        return hashID(kernelMeta.mS, kernelMeta.mD, kernelMeta.mInterleaved, kernelMeta.mUnrollStep,
            kernelMeta.mFP32Accumulation, kernelMeta.mFlashAttention, kernelMeta.mWarpSpecialization,
            kernelMeta.mAlibiSupported, kernelMeta.mAttentionMaskType, kernelMeta.mAttentionInputLayout,
            kernelMeta.mTiled, kernelMeta.mEnableQKTanhScale);
    }

    // FMHA runner.
    void run(Fused_multihead_attention_params_v2& params, Launch_params& launch_params, cudaStream_t stream) const
    {
        bool forceUnroll = useForceUnroll(params, launch_params);
        auto const findIter = mFunctions.find(hashFromParams(params, launch_params));

        // Add debug info when kernels are not found.
        TLLM_CHECK_WITH_INFO(findIter != mFunctions.end(),
            "FMHA kernels are not found (kernel meta info: %d %d %d %d %d %d %d %d %d %d %d %d) !",
            launch_params.kernel_s, params.d, launch_params.interleaved, forceUnroll, launch_params.force_fp32_acc,
            launch_params.flash_attention, launch_params.warp_specialization, !launch_params.useKernelWithoutAlibi,
            static_cast<int>(launch_params.attention_mask_type), static_cast<int>(launch_params.attention_input_layout),
            launch_params.granular_tiling, launch_params.enableQKTanhScale);

        auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;

        void* kernelParams[] = {&params, nullptr};

        if (!forceUnroll)
        {
            cuErrCheck(mDriver->cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                           kernelMeta.mSharedMemBytes, stream, kernelParams, nullptr),
                mDriver);
        } // forceunroll = true for flash attention kernels
        else if (mSM == kSM_90 && launch_params.flash_attention && launch_params.warp_specialization)
        {
            dim3 block_size;

            if (launch_params.dynamic_scheduler)
            {
                // Get the max total M steps
                size_t m_steps = size_t((params.s + kernelMeta.mUnrollStep - 1) / kernelMeta.mUnrollStep);
                m_steps = size_t((m_steps + NUM_COMPUTE_GROUPS - 1) / NUM_COMPUTE_GROUPS);
                params.num_tiles_per_head = static_cast<uint32_t>(m_steps);
                params.num_tiles = static_cast<uint32_t>(m_steps * params.b * params.h);

                block_size.y = std::min(static_cast<int>(params.num_tiles), launch_params.multi_processor_count);
                // 2 * bytes_per_elt stands for kv cache and bytes_per_elt bytes per element.
                size_t size_in_bytes = params.b * params.h * params.s * params.d * 2 * get_size_in_bytes(mDataType);
                params.use_balanced_scheduling = launch_params.attention_mask_type == ContextAttentionMaskType::CAUSAL
                    && size_in_bytes <= launch_params.device_l2_cache_size;

                block_size.x = 1;
                block_size.y = std::min(static_cast<int>(params.num_tiles), launch_params.multi_processor_count);
            }
            else
            {
                // Note that this path won't be used. will be dropped later.
                // tricks for launching warp-specialized flash attention kernels on Hopper
                block_size.y = std::min(params.b * params.h, launch_params.multi_processor_count);

                // distribute m steps to multiple blocks (fully utilize SMs)
                // block.x = blocks that handle single head, block.y = blocks that handle different heads
                size_t sms_per_head = (launch_params.multi_processor_count) / block_size.y;
                size_t m_steps = size_t((params.s + kernelMeta.mUnrollStep * NUM_COMPUTE_GROUPS - 1)
                    / kernelMeta.mUnrollStep * NUM_COMPUTE_GROUPS);

                // 2 * size_per_element stands for kv cache.
                size_t size_in_bytes = block_size.y * params.s * params.d * 2 * get_size_in_bytes(mDataType);
                if (size_in_bytes <= launch_params.device_l2_cache_size)
                {
                    // strategy 1: limit to only 1 wave
                    block_size.x = std::min(m_steps / NUM_COMPUTE_GROUPS, sms_per_head);
                }
                else
                {
                    // strategy 2: fully unroll the q loops (contiguous blocks handle all q loops)
                    block_size.x = m_steps / NUM_COMPUTE_GROUPS;
                }
            }

            cuErrCheck(mDriver->cuLaunchKernel(func, block_size.x, block_size.y, block_size.z,
                           kernelMeta.mThreadsPerCTA, 1, 1, kernelMeta.mSharedMemBytes, stream, kernelParams, nullptr),
                mDriver);
        }
        else
        { // forceunroll = true for flash attention kernels
            int unroll = kernelMeta.mS / kernelMeta.mUnrollStep;
            TLLM_CHECK_WITH_INFO(kernelMeta.mS == kernelMeta.mUnrollStep * unroll, "Wrong launching sequence length");
            // flash attention supports any sequence length, so we runtime s here
            if (launch_params.flash_attention)
            {
                unroll = (params.s + kernelMeta.mUnrollStep - 1) / kernelMeta.mUnrollStep;
            }

            // on Hopper non-flash-attention, we still launch blocks (h, b, steps)
            if (mSM == kSM_90 && !launch_params.flash_attention)
            {
                cuErrCheck(mDriver->cuLaunchKernel(func, params.h, params.b, unroll, kernelMeta.mThreadsPerCTA, 1, 1,
                               kernelMeta.mSharedMemBytes, stream, kernelParams, nullptr),
                    mDriver);
            } // on Ampere/Ada/Volta flash attention, we launch blocks (steps, h, b)
            else
            {
                cuErrCheck(mDriver->cuLaunchKernel(func, unroll, params.h, params.b, kernelMeta.mThreadsPerCTA, 1, 1,
                               kernelMeta.mSharedMemBytes, stream, kernelParams, nullptr),
                    mDriver);
            }
        }
    }

    // Check if any kernels support the attention types during building the engines.
    // Runtime parameters will be set to 0.
    bool checkIfKernelExist(MHARunnerFixedParams params) const override
    {
        uint64_t id = hashID(0, params.headSize, 0, 0, params.forceFp32Acc, false, false, false,
            static_cast<int>(params.attentionMaskType), static_cast<int>(params.attentionInputLayout), false,
            params.qkTanhScale != 0.f);
        auto const findIter = std::find_if(mFunctions.begin(), mFunctions.end(), KernelExistPredicate(id));
        return findIter != mFunctions.end();
    }

    void getStepSize(uint32_t& out_step_q, uint32_t& out_step_kv, Fused_multihead_attention_params_v2 const& params,
        Launch_params const& launch_params) const override
    {
        auto const findIter = mFunctions.find(hashFromParams(params, launch_params));
        TLLM_CHECK_WITH_INFO(findIter != mFunctions.end(),
            "FMHA kernels are not found (kernel meta info: %d %d %d %d %d %d %d %d %d %d %d) !", launch_params.kernel_s,
            params.d, launch_params.interleaved, launch_params.force_fp32_acc, launch_params.flash_attention,
            launch_params.warp_specialization, !launch_params.useKernelWithoutAlibi,
            static_cast<int>(launch_params.attention_mask_type), static_cast<int>(launch_params.attention_input_layout),
            launch_params.granular_tiling, launch_params.enableQKTanhScale);

        auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        out_step_q = kernelMeta.mStepQ;
        out_step_kv = kernelMeta.mStepKV;
    }

private:
    bool useForceUnroll(Fused_multihead_attention_params_v2 const& params, Launch_params const& launch_params) const
    {
        bool forceUnroll = launch_params.force_unroll;
        // Non-flash-attention path.
        if (!forceUnroll && !launch_params.ignore_b1opt && mSM >= kSM_80)
        {
            const struct
            {
                unsigned int mSM;
                Data_type mDataType;
                int mS;
                int mD;
                int mMaxBatchHead;
            } unrollList[] = {
#if CUDA_VERSION >= 11080
                {kSM_90, DATA_TYPE_FP16, 64, 64, 256},
                {kSM_90, DATA_TYPE_FP16, 128, 64, 128},
                {kSM_90, DATA_TYPE_FP16, 256, 64, 128},
                {kSM_90, DATA_TYPE_FP16, 384, 64, 64},
                {kSM_90, DATA_TYPE_FP16, 512, 64, 64},
                {kSM_90, DATA_TYPE_BF16, 64, 64, 256},
                {kSM_90, DATA_TYPE_BF16, 128, 64, 128},
                {kSM_90, DATA_TYPE_BF16, 256, 64, 128},
                {kSM_90, DATA_TYPE_BF16, 384, 64, 64},
                {kSM_90, DATA_TYPE_BF16, 512, 64, 64}
#endif
            };
            for (unsigned int i = 0u; i < sizeof(unrollList) / sizeof(unrollList[0]); ++i)
            {
                if (mSM == unrollList[i].mSM && mDataType == unrollList[i].mDataType
                    && launch_params.kernel_s == unrollList[i].mS && params.d == unrollList[i].mD
                    && params.b * params.h <= unrollList[i].mMaxBatchHead)
                {
                    forceUnroll = true;
                    break;
                }
            }
        }

        return forceUnroll;
    }

    uint64_t hashFromParams(Fused_multihead_attention_params_v2 const& params, Launch_params const& launch_params) const
    {
        bool forceUnroll = useForceUnroll(params, launch_params);
        return hashID(launch_params.kernel_s, params.d, launch_params.interleaved, forceUnroll,
            launch_params.force_fp32_acc, launch_params.flash_attention, launch_params.warp_specialization,
            !launch_params.useKernelWithoutAlibi, static_cast<int>(launch_params.attention_mask_type),
            static_cast<int>(launch_params.attention_input_layout), launch_params.granular_tiling,
            launch_params.enableQKTanhScale);
    }
};

using FusedMHAKernelFactoryV2 = TFusedMHAKernelFactory<FusedMultiHeadAttentionXMMAKernelV2>;

inline FusedMultiHeadAttentionXMMAKernelV2 const* getXMMAKernelsV2(Data_type type, unsigned int sm)
{
    return FusedMHAKernelFactoryV2::Get().getXMMAKernels(
        sMhaKernelMetaInfosV2, sizeof(sMhaKernelMetaInfosV2) / sizeof(sMhaKernelMetaInfosV2[0]), type, sm);
}

} // namespace kernels
} // namespace tensorrt_llm
