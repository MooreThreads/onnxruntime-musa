#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaCastInt32ToFloatKernel(const int32_t* input,
                                             float* output, int64_t count,
                                             musaStream_t stream);

musaError_t LaunchMusaCastInt64ToFloatKernel(const int64_t* input,
                                             float* output, int64_t count,
                                             musaStream_t stream);

musaError_t LaunchMusaCastKernel(const void* input, void* output,
                                 int32_t src_type, int32_t dst_type,
                                 int64_t count, musaStream_t stream);
