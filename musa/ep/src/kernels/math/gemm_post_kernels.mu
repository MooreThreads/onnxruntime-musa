#include "gemm_post_kernels.h"

#include <math.h>
#include <stdint.h>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 4096;

int BlocksForCount(int64_t count) {
  int64_t blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (blocks > kMaxBlocks) {
    blocks = kMaxBlocks;
  }
  return static_cast<int>(blocks);
}

__device__ __forceinline__ void ResolveBroadcastIndices(
    int64_t index, const MusaBroadcastParams& params, int64_t& lhs_index,
    int64_t& rhs_index) {
  lhs_index = 0;
  rhs_index = 0;
  int64_t remaining = index;
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    const int64_t coord = remaining / params.output_strides[dim];
    remaining -= coord * params.output_strides[dim];
    lhs_index += coord * params.lhs_strides[dim];
    rhs_index += coord * params.rhs_strides[dim];
  }
}

__device__ __forceinline__ float UnaryValue(float x, MusaUnaryOp op,
                                            float alpha) {
  switch (op) {
    case MusaUnaryOp::Relu:
      return x > 0.0f ? x : 0.0f;
    case MusaUnaryOp::LeakyRelu:
      return x >= 0.0f ? x : alpha * x;
    case MusaUnaryOp::Sqrt:
      return sqrtf(x);
    case MusaUnaryOp::Reciprocal:
      return 1.0f / x;
    case MusaUnaryOp::Neg:
      return -x;
    case MusaUnaryOp::Log:
      return logf(x);
    case MusaUnaryOp::Tanh:
      return tanhf(x);
    case MusaUnaryOp::Sigmoid:
      return 1.0f / (1.0f + expf(-x));
    case MusaUnaryOp::Abs:
      return fabsf(x);
    case MusaUnaryOp::Erf:
      return erff(x);
  }
  return x;
}

__global__ void GemmPostFloatKernel(float* output,
                                    const float* bias,
                                    MusaBroadcastParams params,
                                    bool has_bias,
                                    float beta,
                                    MusaUnaryOp activation,
                                    bool has_activation,
                                    float activation_alpha) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements;
       index += total_threads) {
    float value = output[index];
    if (has_bias) {
      int64_t output_index = 0;
      int64_t bias_index = 0;
      ResolveBroadcastIndices(index, params, output_index, bias_index);
      value += beta * bias[bias_index];
    }
    if (has_activation) {
      value = UnaryValue(value, activation, activation_alpha);
    }
    output[index] = value;
  }
}

}  // namespace

musaError_t LaunchMusaGemmPostFloatKernel(float* output,
                                          const float* bias,
                                          MusaBroadcastParams params,
                                          bool has_bias,
                                          float beta,
                                          MusaUnaryOp activation,
                                          bool has_activation,
                                          float activation_alpha,
                                          musaStream_t stream) {
  if (params.total_elements == 0 || (!has_bias && !has_activation)) {
    return musaSuccess;
  }
  GemmPostFloatKernel<<<BlocksForCount(params.total_elements),
                        kThreadsPerBlock, 0, stream>>>(
      output, bias, params, has_bias, beta, activation, has_activation,
      activation_alpha);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}
