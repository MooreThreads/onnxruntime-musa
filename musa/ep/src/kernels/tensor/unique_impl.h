#pragma once

#include <musa_runtime.h>

#include <cstdint>

#include "shared_inc/device_kernel_types.h"

int UniqueBlockCount(int64_t total_elements);

musaError_t LaunchMusaUniqueCountKernel(const void* input, int64_t count,
                                        int* first_flags, int* block_counts,
                                        MusaElementType elem_type,
                                        musaStream_t stream);

musaError_t LaunchMusaUniqueOutputKernel(
    const void* input, int64_t input_count, int64_t unique_count,
    const int* first_flags, void* values, int64_t* indices,
    int64_t* inverse_indices, int64_t* counts, int sorted,
    MusaElementType elem_type, musaStream_t stream);
