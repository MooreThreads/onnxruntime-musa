#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaConcatCopies(void* output, const void* const* inputs,
                                   const int64_t* input_axis_dims,
                                   int64_t input_count, int64_t outer,
                                   int64_t inner, int64_t output_axis,
                                   int32_t element_size,
                                   musaStream_t stream);
