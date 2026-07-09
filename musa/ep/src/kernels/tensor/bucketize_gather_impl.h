#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaBucketizeGatherKernel(
    const void* table, const void* indices, void* output, int32_t element_size,
    int32_t index_element_size, int64_t indices_count, int64_t table_rows,
    int64_t block_size, int64_t modulus, int64_t offset,
    float greater_threshold, musaStream_t stream);
