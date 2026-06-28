#pragma once

#include <musa_runtime.h>

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaPReluKernel(const void* input, const void* slope,
                                  void* output, MusaBroadcastParams params,
                                  int32_t elem_type, musaStream_t stream);
