#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaTopKKernel(const void* input, void* values,
                                 int64_t* indices, MusaTopKParams params,
                                 MusaElementType elem_type,
                                 musaStream_t stream);
