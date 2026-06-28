#pragma once

#include <musa_runtime.h>

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaScatterElementsKernel(
    void* output, const void* indices, const void* updates,
    int32_t element_size, int32_t index_element_size, int32_t elem_type,
    MusaScatterElementsParams params, musaStream_t stream);
