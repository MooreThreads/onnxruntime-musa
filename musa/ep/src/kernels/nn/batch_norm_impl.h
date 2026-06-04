#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaBatchNormalizationFloatKernel(const float* input,
                                                    const float* scale,
                                                    const float* bias,
                                                    const float* mean,
                                                    const float* variance,
                                                    float* output,
                                                    MusaBatchNormParams params,
                                                    musaStream_t stream);

musaError_t LaunchMusaBatchNormalizationKernel(const void* input,
                                               const void* scale,
                                               const void* bias,
                                               const void* mean,
                                               const void* variance,
                                               void* output,
                                               MusaBatchNormParams params,
                                               MusaElementType elem_type,
                                               musaStream_t stream);
