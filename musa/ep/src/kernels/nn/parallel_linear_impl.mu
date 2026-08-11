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

template <int BranchCount>
__global__ void ParallelLinearPostDirectKernel(
    const float* merged, float* output0, float* output1, float* output2,
    const float* bias0, const float* bias1, const float* bias2, int64_t rows,
    int64_t branch_width, MusaUnaryOp activation, bool has_activation,
    float activation_alpha) {
  const int64_t total = rows * BranchCount * branch_width;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += stride) {
    const int64_t branch_elements = rows * branch_width;
    const int64_t branch = index / branch_elements;
    const int64_t local = index - branch * branch_elements;
    const int64_t row = local / branch_width;
    const int64_t column = local - row * branch_width;
    float* output = branch == 0 ? output0 : (branch == 1 ? output1 : output2);
    const float* bias = branch == 0 ? bias0 : (branch == 1 ? bias1 : bias2);
    float value = merged[row * BranchCount * branch_width +
                         branch * branch_width + column];
    if (bias != nullptr) {
      value += bias[column];
    }
    if (has_activation) {
      value = ApplyActivation(value, activation, activation_alpha);
    }
    output[local] = value;
  }
}

__global__ void ParallelLinearGatedMlpPostKernel(
    const float* merged, float* output, const float* gate_bias,
    const float* up_bias, int64_t rows, int64_t branch_width) {
  const int64_t total = rows * branch_width;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += stride) {
    const int64_t row = index / branch_width;
    const int64_t column = index - row * branch_width;
    const int64_t merged_offset = row * 2 * branch_width + column;
    float gate = merged[merged_offset];
    float up = merged[merged_offset + branch_width];
    if (gate_bias != nullptr) {
      gate += gate_bias[column];
    }
    if (up_bias != nullptr) {
      up += up_bias[column];
    }
    output[index] =
        gate * ApplyActivation(gate, MusaUnaryOp::Sigmoid, 0.0f) * up;
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

musaError_t LaunchParallelLinearPostDirectFloatKernel(
    const float* merged, float* output0, float* output1, float* output2,
    const float* bias0, const float* bias1, const float* bias2, int64_t rows,
    int64_t branch_count, int64_t branch_width, MusaUnaryOp activation,
    bool has_activation, float activation_alpha, musaStream_t stream) {
  const int64_t total = rows * branch_count * branch_width;
  if (total == 0) {
    return musaSuccess;
  }
  if (branch_count == 2) {
    ParallelLinearPostDirectKernel<2>
        <<<BlocksForCount(total), kThreadsPerBlock, 0, stream>>>(
            merged, output0, output1, nullptr, bias0, bias1, nullptr, rows,
            branch_width, activation, has_activation, activation_alpha);
  } else if (branch_count == 3) {
    ParallelLinearPostDirectKernel<3>
        <<<BlocksForCount(total), kThreadsPerBlock, 0, stream>>>(
            merged, output0, output1, output2, bias0, bias1, bias2, rows,
            branch_width, activation, has_activation, activation_alpha);
  } else {
    return musaErrorInvalidValue;
  }
  return musaGetLastError();
}

musaError_t LaunchParallelLinearGatedMlpPostFloatKernel(
    const float* merged, float* output, const float* gate_bias,
    const float* up_bias, int64_t rows, int64_t branch_width,
    musaStream_t stream) {
  const int64_t total = rows * branch_width;
  if (total == 0) {
    return musaSuccess;
  }
  ParallelLinearGatedMlpPostKernel<<<BlocksForCount(total), kThreadsPerBlock, 0,
                                     stream>>>(merged, output, gate_bias,
                                               up_bias, rows, branch_width);
  return musaGetLastError();
}
