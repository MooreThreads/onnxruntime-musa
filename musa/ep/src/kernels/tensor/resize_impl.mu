#include "shared_inc/musa_kernel_common.mu.h"
#include "tensor/resize_impl.h"

namespace {

__global__ void ResizeNearest2DFloatKernel(
    const float* input, float* output, int64_t total_elements, int64_t channels,
    int64_t input_height, int64_t input_width, int64_t output_height,
    int64_t output_width, float scale_height, float scale_width) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < total_elements;
       output_index += total_threads) {
    int64_t remaining = output_index;
    const int64_t output_x = remaining % output_width;
    remaining /= output_width;
    const int64_t output_y = remaining % output_height;
    remaining /= output_height;
    const int64_t channel = remaining % channels;
    const int64_t batch = remaining / channels;

    // ORT evaluates the nearest neighbour coordinate as `float(dim) / scale`
    // (upsample.cc: SetupUpsampleNearestInputMappings). __fdiv_rn keeps that
    // division IEEE-round-to-nearest on the device; the default lowering picks
    // the approximate div.full.f32, which turns an exactly integral ratio such
    // as 1 / (3.0f / 9.0f) into 2.9999998f and truncates one sample short.
    int64_t input_x = static_cast<int64_t>(
        __fdiv_rn(static_cast<float>(output_x), scale_width));
    int64_t input_y = static_cast<int64_t>(
        __fdiv_rn(static_cast<float>(output_y), scale_height));
    input_x = input_x < input_width ? input_x : input_width - 1;
    input_y = input_y < input_height ? input_y : input_height - 1;
    const int64_t input_index =
        ((batch * channels + channel) * input_height + input_y) * input_width +
        input_x;
    output[output_index] = input[input_index];
  }
}

// Mirrors the coordinate transformation used by the ORT CPU Resize kernel
// (GetOriginalCoordinateFromResizedCoordinate in upsamplebase.h).
// Every division goes through __fdiv_rn so the coordinates are IEEE-rounded
// like the host kernel instead of the approximate div.full.f32 the default
// device lowering emits; that difference is visible in the interpolation
// weights and, near an integer boundary, in the chosen input sample.
__device__ __forceinline__ float ResizeOriginalCoordinate(
    int32_t coordinate_mode, int64_t output_index, float scale,
    int64_t output_size, int64_t input_size) {
  const float resized = static_cast<float>(output_index);
  switch (coordinate_mode) {
    case kMusaResizePytorchHalfPixel:
      return output_size > 1 ? __fdiv_rn(resized + 0.5f, scale) - 0.5f : 0.0f;
    case kMusaResizeAlignCorners:
      return output_size == 1
                 ? 0.0f
                 : __fdiv_rn(resized * (static_cast<float>(input_size) - 1.0f),
                             static_cast<float>(output_size) - 1.0f);
    case kMusaResizeAsymmetric:
      return __fdiv_rn(resized, scale);
    default:  // kMusaResizeHalfPixel
      return __fdiv_rn(resized + 0.5f, scale) - 0.5f;
  }
}

// Mirrors SetupUpsampleBilinear/SetupUpsampleTrilinear: when the scale is
// exactly 1 the coordinate is the index itself; otherwise it is transformed
// and clamped to the input range. The two neighbours always collapse to the
// same sample at the last index, where both weights become 0.5.
__device__ __forceinline__ void ResizeAxisWeights(
    int32_t coordinate_mode, int64_t index, float scale, int64_t input_size,
    int64_t output_size, int64_t& lower, int64_t& upper, float& weight1,
    float& weight2) {
  float original = scale == 1.0f
                       ? static_cast<float>(index)
                       : ResizeOriginalCoordinate(coordinate_mode, index, scale,
                                                  output_size, input_size);
  original = fmaxf(0.0f, fminf(original, static_cast<float>(input_size - 1)));
  lower = static_cast<int64_t>(original);
  lower = lower < input_size - 1 ? lower : input_size - 1;
  upper = lower + 1 < input_size - 1 ? lower + 1 : input_size - 1;
  weight1 = fabsf(original - static_cast<float>(lower));
  weight2 = fabsf(original - static_cast<float>(upper));
  if (lower == upper) {
    weight1 = 0.5f;
    weight2 = 0.5f;
  }
}

__global__ void ResizeLinear2DFloatKernel(
    const float* input, float* output, int64_t total_elements, int64_t channels,
    int64_t input_height, int64_t input_width, int64_t output_height,
    int64_t output_width, float scale_height, float scale_width,
    int32_t coordinate_mode) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < total_elements;
       output_index += total_threads) {
    int64_t remaining = output_index;
    const int64_t output_x = remaining % output_width;
    remaining /= output_width;
    const int64_t output_y = remaining % output_height;
    remaining /= output_height;
    const int64_t channel = remaining % channels;
    const int64_t batch = remaining / channels;

    int64_t y1 = 0;
    int64_t y2 = 0;
    int64_t x1 = 0;
    int64_t x2 = 0;
    float dy1 = 0.0f;
    float dy2 = 0.0f;
    float dx1 = 0.0f;
    float dx2 = 0.0f;
    ResizeAxisWeights(coordinate_mode, output_y, scale_height, input_height,
                      output_height, y1, y2, dy1, dy2);
    ResizeAxisWeights(coordinate_mode, output_x, scale_width, input_width,
                      output_width, x1, x2, dx1, dx2);

    const float* plane =
        input + (batch * channels + channel) * input_height * input_width;
    const float x11 = plane[y1 * input_width + x1];
    const float x21 = plane[y1 * input_width + x2];
    const float x12 = plane[y2 * input_width + x1];
    const float x22 = plane[y2 * input_width + x2];
    output[output_index] =
        dx2 * dy2 * x11 + dx1 * dy2 * x21 + dx2 * dy1 * x12 + dx1 * dy1 * x22;
  }
}

__global__ void ResizeLinear3DFloatKernel(
    const float* input, float* output, int64_t total_elements, int64_t channels,
    int64_t input_depth, int64_t input_height, int64_t input_width,
    int64_t output_depth, int64_t output_height, int64_t output_width,
    float scale_depth, float scale_height, float scale_width,
    int32_t coordinate_mode) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < total_elements;
       output_index += total_threads) {
    int64_t remaining = output_index;
    const int64_t output_x = remaining % output_width;
    remaining /= output_width;
    const int64_t output_y = remaining % output_height;
    remaining /= output_height;
    const int64_t output_z = remaining % output_depth;
    remaining /= output_depth;
    const int64_t channel = remaining % channels;
    const int64_t batch = remaining / channels;

    int64_t z1 = 0;
    int64_t z2 = 0;
    int64_t y1 = 0;
    int64_t y2 = 0;
    int64_t x1 = 0;
    int64_t x2 = 0;
    float dz1 = 0.0f;
    float dz2 = 0.0f;
    float dy1 = 0.0f;
    float dy2 = 0.0f;
    float dx1 = 0.0f;
    float dx2 = 0.0f;
    ResizeAxisWeights(coordinate_mode, output_z, scale_depth, input_depth,
                      output_depth, z1, z2, dz1, dz2);
    ResizeAxisWeights(coordinate_mode, output_y, scale_height, input_height,
                      output_height, y1, y2, dy1, dy2);
    ResizeAxisWeights(coordinate_mode, output_x, scale_width, input_width,
                      output_width, x1, x2, dx1, dx2);

    const int64_t plane_stride = input_height * input_width;
    const float* volume =
        input + (batch * channels + channel) * input_depth * plane_stride;
    const float x111 = volume[z1 * plane_stride + y1 * input_width + x1];
    const float x211 = volume[z1 * plane_stride + y1 * input_width + x2];
    const float x121 = volume[z1 * plane_stride + y2 * input_width + x1];
    const float x221 = volume[z1 * plane_stride + y2 * input_width + x2];
    const float x112 = volume[z2 * plane_stride + y1 * input_width + x1];
    const float x212 = volume[z2 * plane_stride + y1 * input_width + x2];
    const float x122 = volume[z2 * plane_stride + y2 * input_width + x1];
    const float x222 = volume[z2 * plane_stride + y2 * input_width + x2];
    output[output_index] = dx2 * dy2 * dz2 * x111 + dx1 * dy2 * dz2 * x211 +
                           dx2 * dy1 * dz2 * x121 + dx1 * dy1 * dz2 * x221 +
                           dx2 * dy2 * dz1 * x112 + dx1 * dy2 * dz1 * x212 +
                           dx2 * dy1 * dz1 * x122 + dx1 * dy1 * dz1 * x222;
  }
}

}  // namespace

musaError_t LaunchMusaResizeNearest2DFloat(
    const float* input, float* output, int64_t batch, int64_t channels,
    int64_t input_height, int64_t input_width, int64_t output_height,
    int64_t output_width, float scale_height, float scale_width,
    musaStream_t stream) {
  const int64_t total_elements =
      batch * channels * output_height * output_width;
  if (total_elements == 0) {
    return musaSuccess;
  }
  ResizeNearest2DFloatKernel<<<BlocksForCount(total_elements), kThreadsPerBlock,
                               0, stream>>>(
      input, output, total_elements, channels, input_height, input_width,
      output_height, output_width, scale_height, scale_width);
  return musaGetLastError();
}

musaError_t LaunchMusaResizeLinear2DFloat(
    const float* input, float* output, int64_t batch, int64_t channels,
    int64_t input_height, int64_t input_width, int64_t output_height,
    int64_t output_width, float scale_height, float scale_width,
    int32_t coordinate_mode, musaStream_t stream) {
  const int64_t total_elements =
      batch * channels * output_height * output_width;
  if (total_elements == 0) {
    return musaSuccess;
  }
  ResizeLinear2DFloatKernel<<<BlocksForCount(total_elements), kThreadsPerBlock,
                              0, stream>>>(
      input, output, total_elements, channels, input_height, input_width,
      output_height, output_width, scale_height, scale_width, coordinate_mode);
  return musaGetLastError();
}

musaError_t LaunchMusaResizeLinear3DFloat(
    const float* input, float* output, int64_t batch, int64_t channels,
    int64_t input_depth, int64_t input_height, int64_t input_width,
    int64_t output_depth, int64_t output_height, int64_t output_width,
    float scale_depth, float scale_height, float scale_width,
    int32_t coordinate_mode, musaStream_t stream) {
  const int64_t total_elements =
      batch * channels * output_depth * output_height * output_width;
  if (total_elements == 0) {
    return musaSuccess;
  }
  ResizeLinear3DFloatKernel<<<BlocksForCount(total_elements), kThreadsPerBlock,
                              0, stream>>>(
      input, output, total_elements, channels, input_depth, input_height,
      input_width, output_depth, output_height, output_width, scale_depth,
      scale_height, scale_width, coordinate_mode);
  return musaGetLastError();
}
