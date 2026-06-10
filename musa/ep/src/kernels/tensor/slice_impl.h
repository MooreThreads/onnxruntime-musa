#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaSliceKernel(const void* input,
                                  void* output,
                                  int32_t element_size,
                                  MusaSliceParams params,
                                  musaStream_t stream);
