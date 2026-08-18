#pragma once

#include <cstdint>

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaSiluKernel(const void* input, void* output, int64_t count,
                                 MusaElementType elem_type,
                                 musaStream_t stream);
