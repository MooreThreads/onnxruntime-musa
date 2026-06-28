#pragma once

#include <musa_runtime.h>

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaAttentionAddBiasKernel(float* qkv, const float* bias,
                                             int64_t total_elements,
                                             int64_t qkv_hidden_size,
                                             musaStream_t stream);

musaError_t LaunchMusaAttentionScoreKernel(const float* qkv,
                                           const int32_t* mask, float* scores,
                                           MusaAttentionParams params,
                                           musaStream_t stream);

musaError_t LaunchMusaAttentionValueKernel(const float* qkv,
                                           const float* scores, float* output,
                                           MusaAttentionParams params,
                                           musaStream_t stream);
