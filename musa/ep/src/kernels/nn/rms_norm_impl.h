#pragma once

#include <musa_runtime.h>

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaRmsNormKernel(const void* input, const float* gamma,
                                    void* output, int64_t rows,
                                    int64_t norm_size, float epsilon,
                                    MusaElementType elem_type,
                                    musaStream_t stream);
