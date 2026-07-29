#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaConv2DFloatKernel(const float* input, const float* weight,
                                        const float* bias, float* output,
                                        MusaConv2DParams params,
                                        musaStream_t stream);
