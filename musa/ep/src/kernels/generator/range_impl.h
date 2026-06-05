#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaRangeKernel(void* output,
                                  MusaRangeParams params,
                                  MusaElementType elem_type,
                                  double start,
                                  double delta,
                                  musaStream_t stream);
