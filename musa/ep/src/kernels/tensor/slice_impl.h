#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaSliceKernel(const void* input,
                                  void* output,
                                  int32_t element_size,
                                  MusaSliceParams params,
                                  musaStream_t stream);

musaError_t LaunchMusaSliceLastAxisKernel(
    const void* input, void* output, int32_t element_size,
    int64_t total_elements, int64_t input_width, int64_t output_width,
    int64_t start, musaStream_t stream);
