#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaTransposeKernel(const void* input,
                                      void* output,
                                      int32_t element_size,
                                      MusaTransposeParams params,
                                      musaStream_t stream);
