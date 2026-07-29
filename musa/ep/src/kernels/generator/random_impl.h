#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaRandomUniformKernel(void* output, int64_t count,
                                          float low, float high, uint64_t seed,
                                          MusaElementType elem_type,
                                          musaStream_t stream);
