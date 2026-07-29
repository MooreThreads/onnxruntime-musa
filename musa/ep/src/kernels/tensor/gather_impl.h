#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaGatherKernel(
    const void* input, const void* indices, void* output, int32_t element_size,
    int32_t index_element_size, int64_t input_block_size, int64_t indices_max,
    int64_t output_block_size, int64_t block_size, int64_t output_count,
    musaStream_t stream);
