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

template <int BranchCount>
__global__ void ParallelLinearPostDirectCopy16Kernel(
    const uint16_t* merged, uint16_t* output0, uint16_t* output1,
    uint16_t* output2, int64_t rows, int64_t branch_width) {
  const int64_t row_branch = blockIdx.x;
  const int64_t row = row_branch / BranchCount;
  if (row >= rows) {
    return;
  }
  const int branch = static_cast<int>(row_branch % BranchCount);
  uint16_t* output =
      branch == 0 ? output0 : (branch == 1 ? output1 : output2);
  const uint16_t* src =
      merged + (row * BranchCount + branch) * branch_width;
  uint16_t* dst = output + row * branch_width;

  constexpr int64_t kElementsPerVector = sizeof(uint4) / sizeof(uint16_t);
  const int64_t vector_count = branch_width / kElementsPerVector;
  const auto* vector_src = reinterpret_cast<const uint4*>(src);
  auto* vector_dst = reinterpret_cast<uint4*>(dst);
  for (int64_t i = threadIdx.x; i < vector_count; i += blockDim.x) {
    vector_dst[i] = vector_src[i];
  }
  for (int64_t i = vector_count * kElementsPerVector + threadIdx.x;
       i < branch_width; i += blockDim.x) {
    dst[i] = src[i];
  }
}

template <int BranchCount>
__global__ void ParallelLinearPostDirectCopyScalarKernel(
    const uint16_t* merged, uint16_t* output0, uint16_t* output1,
    uint16_t* output2, int64_t rows, int64_t branch_width) {
  const int64_t row_branch = blockIdx.x;
  const int64_t row = row_branch / BranchCount;
  if (row >= rows) {
    return;
  }
  const int branch = static_cast<int>(row_branch % BranchCount);
  uint16_t* output =
      branch == 0 ? output0 : (branch == 1 ? output1 : output2);
  const uint16_t* src =
      merged + (row * BranchCount + branch) * branch_width;
  uint16_t* dst = output + row * branch_width;
  for (int64_t i = threadIdx.x; i < branch_width; i += blockDim.x) {
    dst[i] = src[i];
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

musaError_t LaunchParallelLinearPostDirectCopy16Kernel(
    const void* merged, void* output0, void* output1, void* output2,
    int64_t rows, int64_t branch_count, int64_t branch_width,
    musaStream_t stream) {
  if (rows == 0 || branch_width == 0) {
    return musaSuccess;
  }
  const int64_t blocks = rows * branch_count;
  if (blocks > INT32_MAX) {
    return musaErrorInvalidValue;
  }
  const bool can_vectorize =
      branch_width % (sizeof(uint4) / sizeof(uint16_t)) == 0 &&
      (reinterpret_cast<uintptr_t>(merged) % alignof(uint4)) == 0 &&
      (reinterpret_cast<uintptr_t>(output0) % alignof(uint4)) == 0 &&
      (reinterpret_cast<uintptr_t>(output1) % alignof(uint4)) == 0 &&
      (branch_count != 3 ||
       (reinterpret_cast<uintptr_t>(output2) % alignof(uint4)) == 0);
  if (branch_count == 2) {
    if (can_vectorize) {
      ParallelLinearPostDirectCopy16Kernel<2>
          <<<static_cast<int>(blocks), kThreadsPerBlock, 0, stream>>>(
              static_cast<const uint16_t*>(merged),
              static_cast<uint16_t*>(output0), static_cast<uint16_t*>(output1),
              nullptr, rows, branch_width);
    } else {
      ParallelLinearPostDirectCopyScalarKernel<2>
          <<<static_cast<int>(blocks), kThreadsPerBlock, 0, stream>>>(
              static_cast<const uint16_t*>(merged),
              static_cast<uint16_t*>(output0), static_cast<uint16_t*>(output1),
              nullptr, rows, branch_width);
    }
  } else if (branch_count == 3) {
    if (can_vectorize) {
      ParallelLinearPostDirectCopy16Kernel<3>
          <<<static_cast<int>(blocks), kThreadsPerBlock, 0, stream>>>(
              static_cast<const uint16_t*>(merged),
              static_cast<uint16_t*>(output0), static_cast<uint16_t*>(output1),
              static_cast<uint16_t*>(output2), rows, branch_width);
    } else {
      ParallelLinearPostDirectCopyScalarKernel<3>
          <<<static_cast<int>(blocks), kThreadsPerBlock, 0, stream>>>(
              static_cast<const uint16_t*>(merged),
              static_cast<uint16_t*>(output0), static_cast<uint16_t*>(output1),
              static_cast<uint16_t*>(output2), rows, branch_width);
    }
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
