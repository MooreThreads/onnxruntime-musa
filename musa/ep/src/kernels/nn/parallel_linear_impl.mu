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
    int64_t rows, int64_t branch_count, int64_t branch_width,
    MusaUnaryOp activation, bool has_activation, float activation_alpha) {
  const int64_t total = rows * branch_count * branch_width;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += stride) {
    const int64_t branch_elements = rows * branch_width;
    const int64_t branch = index / branch_elements;
    const int64_t local = index - branch * branch_elements;
    const int64_t row = local / branch_width;
    const int64_t column = local - row * branch_width;
    float value = merged[row * branch_count * branch_width +
                         branch * branch_width + column];
    if (biases[branch] != nullptr) {
      value += biases[branch][column];
    }
    if (has_activation) {
      value = ApplyActivation(value, activation, activation_alpha);
    }
    outputs[branch][local] = value;
  }
}

}  // namespace

musaError_t LaunchParallelLinearPostFloatKernel(
    const float* merged, float* const* outputs, const float* const* biases,
    int64_t rows, int64_t branch_count, int64_t branch_width,
    MusaUnaryOp activation, bool has_activation, float activation_alpha,
    musaStream_t stream) {
  const int64_t total = rows * branch_count * branch_width;
  if (total == 0) {
    return musaSuccess;
  }
  ParallelLinearPostKernel<<<BlocksForCount(total), kThreadsPerBlock, 0,
                             stream>>>(merged, outputs, biases, rows,
                                       branch_count, branch_width, activation,
                                       has_activation, activation_alpha);
  return musaGetLastError();
}
