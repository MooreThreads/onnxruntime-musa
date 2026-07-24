#include <math.h>
#include <stdint.h>

#include "parallel_linear_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ float ApplyActivation(float value, MusaUnaryOp op,
                                                 float alpha) {
  switch (op) {
    case MusaUnaryOp::Relu:
      return value > 0.0f ? value : 0.0f;
    case MusaUnaryOp::LeakyRelu:
      return value >= 0.0f ? value : alpha * value;
    case MusaUnaryOp::Tanh:
      return tanhf(value);
    case MusaUnaryOp::Sigmoid:
      return 1.0f / (1.0f + expf(-value));
    default:
      return value;
  }
}

__global__ void ParallelLinearPostKernel(
    const float* merged, float* const* outputs, const float* const* biases,
    const int64_t* branch_widths, const int64_t* branch_offsets, int64_t rows,
    int64_t branch_count, int64_t total_width, MusaUnaryOp activation,
    bool has_activation, float activation_alpha) {
  const int64_t total = rows * total_width;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += stride) {
    const int64_t row = index / total_width;
    const int64_t merged_column = index - row * total_width;
    int64_t branch = 0;
    for (; branch < branch_count - 1; ++branch) {
      const int64_t begin = branch_offsets[branch];
      const int64_t end = begin + branch_widths[branch];
      if (merged_column >= begin && merged_column < end) {
        break;
      }
    }
    const int64_t column = merged_column - branch_offsets[branch];
    const int64_t branch_width = branch_widths[branch];
    float value = merged[index];
    if (biases[branch] != nullptr) {
      value += biases[branch][column];
    }
    if (has_activation) {
      value = ApplyActivation(value, activation, activation_alpha);
    }
    outputs[branch][row * branch_width + column] = value;
  }
}

}  // namespace

musaError_t LaunchParallelLinearPostFloatKernel(
    const float* merged, float* const* outputs, const float* const* biases,
    const int64_t* branch_widths, const int64_t* branch_offsets, int64_t rows,
    int64_t branch_count, int64_t total_width, MusaUnaryOp activation,
    bool has_activation, float activation_alpha, musaStream_t stream) {
  const int64_t total = rows * total_width;
  if (total == 0) {
    return musaSuccess;
  }
  ParallelLinearPostKernel<<<BlocksForCount(total), kThreadsPerBlock, 0,
                             stream>>>(merged, outputs, biases, branch_widths,
                                       branch_offsets, rows, branch_count,
                                       total_width, activation, has_activation,
                                       activation_alpha);
  return musaGetLastError();
}
