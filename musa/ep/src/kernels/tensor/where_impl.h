#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaWhereKernel(const uint8_t* condition,
                                  const void* x,
                                  const void* y,
                                  void* output,
                                  int32_t element_size,
                                  MusaWhereParams params,
                                  musaStream_t stream);
