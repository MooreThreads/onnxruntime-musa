#pragma once

#include <musa_runtime.h>

#include <cstdint>

// Coordinate transformation modes shared by resize.cc (host) and
// resize_impl.mu (device). The numeric values must stay in sync with the
// ONNX attribute strings mapped in resize.cc.
enum MusaResizeCoordinateMode : int32_t {
  kMusaResizeHalfPixel = 0,
  kMusaResizePytorchHalfPixel = 1,
  kMusaResizeAlignCorners = 2,
  kMusaResizeAsymmetric = 3,
};

musaError_t LaunchMusaResizeNearest2DFloat(
    const float* input, float* output, int64_t batch, int64_t channels,
    int64_t input_height, int64_t input_width, int64_t output_height,
    int64_t output_width, float scale_height, float scale_width,
    musaStream_t stream);

musaError_t LaunchMusaResizeLinear2DFloat(
    const float* input, float* output, int64_t batch, int64_t channels,
    int64_t input_height, int64_t input_width, int64_t output_height,
    int64_t output_width, float scale_height, float scale_width,
    int32_t coordinate_mode, musaStream_t stream);

musaError_t LaunchMusaResizeLinear3DFloat(
    const float* input, float* output, int64_t batch, int64_t channels,
    int64_t input_depth, int64_t input_height, int64_t input_width,
    int64_t output_depth, int64_t output_height, int64_t output_width,
    float scale_depth, float scale_height, float scale_width,
    int32_t coordinate_mode, musaStream_t stream);
