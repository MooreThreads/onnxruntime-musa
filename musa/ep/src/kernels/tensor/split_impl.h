#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaSplitCopies(const void* input, void* const* outputs,
                                  const int64_t* split_sizes,
                                  int64_t output_count, int64_t outer,
                                  int64_t inner, int64_t input_axis,
                                  int32_t element_size,
                                  musaStream_t stream);
