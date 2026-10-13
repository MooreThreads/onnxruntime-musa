#pragma once

#include <musa_runtime.h>

#include <cstdint>

musaError_t LaunchMusaResizeNearest2DFloat(
    const float* input, float* output, int64_t batch, int64_t channels,
    int64_t input_height, int64_t input_width, int64_t output_height,
    int64_t output_width, float scale_height, float scale_width,
    musaStream_t stream);
