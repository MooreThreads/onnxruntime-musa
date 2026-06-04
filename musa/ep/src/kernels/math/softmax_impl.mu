#include "math/softmax_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void SoftmaxFloatKernel(const float* input,
                                   float* output,
                                   int64_t rows,
                                   int64_t dim,
                                   int64_t inner) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t row_index = thread_id; row_index < rows; row_index += total_threads) {
    const int64_t outer_index = row_index / inner;
    const int64_t inner_index = row_index - outer_index * inner;
    const int64_t base = outer_index * dim * inner + inner_index;

    float max_value = -INFINITY;
    for (int64_t d = 0; d < dim; ++d) {
      const float value = input[base + d * inner];
      max_value = value > max_value ? value : max_value;
    }

    float sum = 0.0f;
    for (int64_t d = 0; d < dim; ++d) {
      const float value = expf(input[base + d * inner] - max_value);
      output[base + d * inner] = value;
      sum += value;
    }

    const float inv_sum = 1.0f / sum;
    for (int64_t d = 0; d < dim; ++d) {
      output[base + d * inner] *= inv_sum;
    }
  }
}

}  // namespace

musaError_t LaunchMusaSoftmaxFloatKernel(const float* input,
                                         float* output,
                                         int64_t outer,
                                         int64_t dim,
                                         int64_t inner,
                                         musaStream_t stream) {
  const int64_t rows = outer * inner;
  if (rows == 0 || dim == 0) {
    return musaSuccess;
  }
  SoftmaxFloatKernel<<<BlocksForCount(rows), kThreadsPerBlock, 0, stream>>>(input, output, rows, dim, inner);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}
