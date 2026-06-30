#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaScatterNDKernel(void* output, const int64_t* indices,
                                      const void* updates, int32_t element_size,
                                      int32_t elem_type,
                                      MusaScatterNDParams params,
                                      musaStream_t stream);
