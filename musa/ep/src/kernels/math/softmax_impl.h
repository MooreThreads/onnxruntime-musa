#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaSoftmaxKernel(const void* input, void* output,
                                    int64_t outer, int64_t dim, int64_t inner,
                                    MusaElementType elem_type,
                                    musaStream_t stream);

musaError_t LaunchMusaSoftmaxFloatKernel(const float* input, float* output,
                                         int64_t outer, int64_t dim,
                                         int64_t inner, musaStream_t stream);
