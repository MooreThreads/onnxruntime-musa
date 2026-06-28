#pragma once

#include <musa_runtime.h>

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaGatherElementsKernel(const void* data,
                                           const void* indices, void* output,
                                           int32_t element_size,
                                           int32_t index_element_size,
                                           MusaGatherElementsParams params,
                                           musaStream_t stream);
