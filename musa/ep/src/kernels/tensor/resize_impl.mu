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

    int64_t input_x = static_cast<int64_t>(output_x / scale_width);
    int64_t input_y = static_cast<int64_t>(output_y / scale_height);
    input_x = input_x < input_width ? input_x : input_width - 1;
    input_y = input_y < input_height ? input_y : input_height - 1;
    const int64_t input_index =
        ((batch * channels + channel) * input_height + input_y) * input_width +
        input_x;
    output[output_index] = input[input_index];
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
