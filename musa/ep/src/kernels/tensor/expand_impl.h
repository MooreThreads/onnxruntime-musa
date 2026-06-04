#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaExpandKernel(const void* input, void* output,
                                   int32_t element_size,
                                   MusaBroadcastParams params,
                                   musaStream_t stream);
