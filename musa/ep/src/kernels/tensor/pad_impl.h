#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaPadKernel(const void* input, void* output,
                                uint64_t constant_value, int32_t element_size,
                                MusaPadParams params, musaStream_t stream);
