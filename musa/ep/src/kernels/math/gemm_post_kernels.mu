#include "gemm_post_kernels.h"
#include "shared_inc/musa_kernel_common.mu.h"

#include <math.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ double GemmPostActivation(double x,
                                                     MusaUnaryOp op,
                                                     float alpha) {
  switch (op) {
    case MusaUnaryOp::Relu:
      return x > 0.0 ? x : 0.0;
    case MusaUnaryOp::LeakyRelu:
      return x >= 0.0 ? x : static_cast<double>(alpha) * x;
    case MusaUnaryOp::Tanh:
      return tanh(x);
    case MusaUnaryOp::Sigmoid:
      return 1.0 / (1.0 + exp(-x));
    default:
      return x;
  }
}

template <typename T>
__global__ void GemmPostKernel(T* output,
                               const T* bias,
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
    double value = MusaScalarToDouble(output[index]);
    if (has_bias) {
      int64_t output_index = 0;
      int64_t bias_index = 0;
      ResolveBroadcastIndices(index, params, output_index, bias_index);
      value += static_cast<double>(beta) * MusaScalarToDouble(bias[bias_index]);
    }
    if (has_activation) {
      value = GemmPostActivation(value, activation, activation_alpha);
    }
    output[index] = MusaScalarFromDouble<T>(value);
  }
}

template <typename T>
musaError_t LaunchGemmPostTyped(void* output,
                                const void* bias,
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
  GemmPostKernel<T><<<BlocksForCount(params.total_elements), kThreadsPerBlock,
                      0, stream>>>(reinterpret_cast<T*>(output),
                                   reinterpret_cast<const T*>(bias), params,
                                   has_bias, beta, activation, has_activation,
                                   activation_alpha);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaGemmPostKernel(void* output,
                                     const void* bias,
                                     MusaBroadcastParams params,
                                     bool has_bias,
                                     float beta,
                                     MusaUnaryOp activation,
                                     bool has_activation,
                                     float activation_alpha,
                                     MusaElementType elem_type,
                                     musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchGemmPostTyped<float>(output, bias, params, has_bias, beta,
                                        activation, has_activation,
                                        activation_alpha, stream);
    case MusaElementType::Double:
      return LaunchGemmPostTyped<double>(output, bias, params, has_bias, beta,
                                         activation, has_activation,
                                         activation_alpha, stream);
    case MusaElementType::Float16:
      return LaunchGemmPostTyped<__half>(output, bias, params, has_bias, beta,
                                         activation, has_activation,
                                         activation_alpha, stream);
    case MusaElementType::BFloat16:
      return LaunchGemmPostTyped<__mt_bfloat16>(
          output, bias, params, has_bias, beta, activation, has_activation,
          activation_alpha, stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaGemmPostFloatKernel(float* output,
                                          const float* bias,
                                          MusaBroadcastParams params,
                                          bool has_bias,
                                          float beta,
                                          MusaUnaryOp activation,
                                          bool has_activation,
                                          float activation_alpha,
                                          musaStream_t stream) {
  return LaunchMusaGemmPostKernel(output, bias, params, has_bias, beta,
                                  activation, has_activation,
                                  activation_alpha, MusaElementType::Float,
                                  stream);
}
