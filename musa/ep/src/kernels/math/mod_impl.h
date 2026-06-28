#pragma once

#include <musa_runtime.h>

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaModKernel(const void* lhs, const void* rhs, void* output,
                                MusaBroadcastParams params, int32_t elem_type,
                                musaStream_t stream);
