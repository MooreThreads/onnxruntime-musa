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
