#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaGatherNDKernel(const void* input, const int64_t* indices,
                                     void* output, int32_t element_size,
                                     MusaGatherNDParams params,
                                     musaStream_t stream);
