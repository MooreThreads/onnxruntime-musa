#pragma once

#include "shared_inc/device_kernel_types.h"

int NonZeroBlockCount(int64_t total_elements);

musaError_t LaunchMusaNonZeroCountKernel(const void* input,
                                         int64_t total_elements,
                                         int* block_counts,
                                         MusaElementType elem_type,
                                         musaStream_t stream);

musaError_t LaunchMusaNonZeroOutputKernel(
    const void* input, const int* prefix_counts, int64_t* output,
    MusaNonZeroParams params, MusaElementType elem_type, musaStream_t stream);
