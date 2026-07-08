#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaMathConcatLogKernel(const float* input, float* output,
                                          int64_t count, float max_value,
                                          float add_value, float scale_value,
                                          musaStream_t stream);
