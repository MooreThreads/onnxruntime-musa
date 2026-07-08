#include "math/math_concat_log_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void MathConcatLogKernel(const float* input,
                                    float* output,
                                    int64_t count,
                                    float max_value,
                                    float add_value,
                                    float scale_value) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    const float clipped = fmaxf(input[index], max_value);
    output[index] = logf(clipped + add_value) * scale_value;
  }
}

}  // namespace

musaError_t LaunchMusaMathConcatLogKernel(const float* input,
                                          float* output,
                                          int64_t count,
                                          float max_value,
                                          float add_value,
                                          float scale_value,
                                          musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  MathConcatLogKernel<<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(
      input, output, count, max_value, add_value, scale_value);
  return musaGetLastError();
}
