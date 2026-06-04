#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaTileKernel(const void* input,
                                 void* output,
                                 int32_t element_size,
                                 MusaTileParams params,
                                 musaStream_t stream);
