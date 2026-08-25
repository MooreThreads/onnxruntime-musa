#include "gemm_post_kernels.h"
#include "shared_inc/musa_kernel_common.mu.h"

#include <math.h>
#include <stdint.h>
#include <type_traits>

namespace {

enum class GemmPostActivationKind : uint8_t {
  Generic = 0,
  Relu = 1,
  LeakyRelu = 2,
  Sigmoid = 3,
};

__device__ __forceinline__ float GemmPostActivationFloat(float x,
                                                          MusaUnaryOp op,
                                                          float alpha) {
  switch (op) {
    case MusaUnaryOp::Relu:
      return x > 0.0f ? x : 0.0f;
    case MusaUnaryOp::LeakyRelu:
      return x >= 0.0f ? x : alpha * x;
    case MusaUnaryOp::Tanh:
      return tanhf(x);
    case MusaUnaryOp::Sigmoid:
      return 1.0f / (1.0f + expf(-x));
    default:
      return x;
  }
}

__device__ __forceinline__ double GemmPostActivationDouble(double x,
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

template <GemmPostActivationKind kActivation>
__device__ __forceinline__ float ApplyGemmPostActivationFloat(
    float value, float alpha) {
  if constexpr (kActivation == GemmPostActivationKind::Relu) {
    return value > 0.0f ? value : 0.0f;
  } else if constexpr (kActivation == GemmPostActivationKind::LeakyRelu) {
    return value >= 0.0f ? value : alpha * value;
  } else {
    // The dispatcher only instantiates this helper for the three specialized
    // activations; keep a neutral fallback for compiler diagnostics.
    return 1.0f / (1.0f + expf(-value));
  }
}

template <GemmPostActivationKind kActivation>
__device__ __forceinline__ double ApplyGemmPostActivationDouble(
    double value, float alpha) {
  if constexpr (kActivation == GemmPostActivationKind::Relu) {
    return value > 0.0 ? value : 0.0;
  } else if constexpr (kActivation == GemmPostActivationKind::LeakyRelu) {
    return value >= 0.0 ? value : static_cast<double>(alpha) * value;
  } else {
    return 1.0 / (1.0 + exp(-value));
  }
}

template <MusaGemmPostBroadcast kBroadcast>
__device__ __forceinline__ int64_t GemmPostBiasIndex(
    int64_t index, const MusaBroadcastParams& params, int64_t inner_size) {
  if constexpr (kBroadcast == MusaGemmPostBroadcast::Scalar) {
    return 0;
  } else if constexpr (kBroadcast == MusaGemmPostBroadcast::LastDim) {
    return index % inner_size;
  } else if constexpr (kBroadcast == MusaGemmPostBroadcast::Full) {
    return index;
  } else {
    int64_t output_index = 0;
    int64_t bias_index = 0;
    ResolveBroadcastIndices(index, params, output_index, bias_index);
    return bias_index;
  }
}

// Generic path for Tanh and any future activation that is not specialized.
template <typename T, MusaGemmPostBroadcast kBroadcast>
__global__ void GemmPostKernelFloatGeneric(T* output,
                                           const T* bias,
                                           MusaBroadcastParams params,
                                           bool has_bias,
                                           float beta,
                                           MusaUnaryOp activation,
                                           bool has_activation,
                                           float activation_alpha,
                                           int64_t inner_size) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements;
       index += total_threads) {
    float value = MusaScalarToFloat(output[index]);
    if (has_bias) {
      const int64_t bias_index =
          GemmPostBiasIndex<kBroadcast>(index, params, inner_size);
      value += beta * MusaScalarToFloat(bias[bias_index]);
    }
    if (has_activation) {
      value = GemmPostActivationFloat(value, activation, activation_alpha);
    }
    output[index] = MusaScalarFromFloat<T>(value);
  }
}

template <MusaGemmPostBroadcast kBroadcast>
__global__ void GemmPostKernelDoubleGeneric(double* output,
                                            const double* bias,
                                            MusaBroadcastParams params,
                                            bool has_bias,
                                            float beta,
                                            MusaUnaryOp activation,
                                            bool has_activation,
                                            float activation_alpha,
                                            int64_t inner_size) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements;
       index += total_threads) {
    double value = output[index];
    if (has_bias) {
      const int64_t bias_index =
          GemmPostBiasIndex<kBroadcast>(index, params, inner_size);
      value += static_cast<double>(beta) * bias[bias_index];
    }
    if (has_activation) {
      value = GemmPostActivationDouble(value, activation, activation_alpha);
    }
    output[index] = value;
  }
}

template <typename T, MusaGemmPostBroadcast kBroadcast,
          GemmPostActivationKind kActivation>
__global__ void GemmPostKernelFloatActivated(T* output,
                                             const T* bias,
                                             MusaBroadcastParams params,
                                             bool has_bias,
                                             float beta,
                                             float activation_alpha,
                                             int64_t inner_size) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements;
       index += total_threads) {
    float value = MusaScalarToFloat(output[index]);
    if (has_bias) {
      const int64_t bias_index =
          GemmPostBiasIndex<kBroadcast>(index, params, inner_size);
      value += beta * MusaScalarToFloat(bias[bias_index]);
    }
    value = ApplyGemmPostActivationFloat<kActivation>(value, activation_alpha);
    output[index] = MusaScalarFromFloat<T>(value);
  }
}

template <MusaGemmPostBroadcast kBroadcast,
          GemmPostActivationKind kActivation>
__global__ void GemmPostKernelDoubleActivated(double* output,
                                              const double* bias,
                                              MusaBroadcastParams params,
                                              bool has_bias,
                                              float beta,
                                              float activation_alpha,
                                              int64_t inner_size) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements;
       index += total_threads) {
    double value = output[index];
    if (has_bias) {
      const int64_t bias_index =
          GemmPostBiasIndex<kBroadcast>(index, params, inner_size);
      value += static_cast<double>(beta) * bias[bias_index];
    }
    value = ApplyGemmPostActivationDouble<kActivation>(value, activation_alpha);
    output[index] = value;
  }
}

template <typename T, MusaGemmPostBroadcast kBroadcast>
__global__ void GemmPostKernelFloatNoActivation(T* output,
                                                const T* bias,
                                                MusaBroadcastParams params,
                                                bool has_bias,
                                                float beta,
                                                int64_t inner_size) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements;
       index += total_threads) {
    float value = MusaScalarToFloat(output[index]);
    if (has_bias) {
      const int64_t bias_index =
          GemmPostBiasIndex<kBroadcast>(index, params, inner_size);
      value += beta * MusaScalarToFloat(bias[bias_index]);
    }
    output[index] = MusaScalarFromFloat<T>(value);
  }
}

template <MusaGemmPostBroadcast kBroadcast>
__global__ void GemmPostKernelDoubleNoActivation(double* output,
                                                 const double* bias,
                                                 MusaBroadcastParams params,
                                                 bool has_bias,
                                                 float beta,
                                                 int64_t inner_size) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements;
       index += total_threads) {
    double value = output[index];
    if (has_bias) {
      const int64_t bias_index =
          GemmPostBiasIndex<kBroadcast>(index, params, inner_size);
      value += static_cast<double>(beta) * bias[bias_index];
    }
    output[index] = value;
  }
}

template <typename T, MusaGemmPostBroadcast kBroadcast>
musaError_t LaunchGemmPostGeneric(void* output,
                                   const void* bias,
                                   MusaBroadcastParams params,
                                   bool has_bias,
                                   float beta,
                                   MusaUnaryOp activation,
                                   bool has_activation,
                                   float activation_alpha,
                                   musaStream_t stream,
                                   int64_t inner_size) {
  if constexpr (std::is_same<T, double>::value) {
    GemmPostKernelDoubleGeneric<kBroadcast><<<
        BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<double*>(output), reinterpret_cast<const double*>(bias),
        params, has_bias, beta, activation, has_activation, activation_alpha,
        inner_size);
  } else {
    GemmPostKernelFloatGeneric<T, kBroadcast><<<
        BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<T*>(output), reinterpret_cast<const T*>(bias), params,
        has_bias, beta, activation, has_activation, activation_alpha,
        inner_size);
  }
  return musaGetLastError();
}

template <typename T, MusaGemmPostBroadcast kBroadcast,
          GemmPostActivationKind kActivation>
musaError_t LaunchGemmPostActivated(void* output,
                                     const void* bias,
                                     MusaBroadcastParams params,
                                     bool has_bias,
                                     float beta,
                                     float activation_alpha,
                                     musaStream_t stream,
                                     int64_t inner_size) {
  if constexpr (std::is_same<T, double>::value) {
    GemmPostKernelDoubleActivated<kBroadcast, kActivation><<<
        BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<double*>(output), reinterpret_cast<const double*>(bias),
        params, has_bias, beta, activation_alpha, inner_size);
  } else {
    GemmPostKernelFloatActivated<T, kBroadcast, kActivation><<<
        BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<T*>(output), reinterpret_cast<const T*>(bias), params,
        has_bias, beta, activation_alpha, inner_size);
  }
  return musaGetLastError();
}

template <typename T, MusaGemmPostBroadcast kBroadcast>
musaError_t LaunchGemmPostNoActivation(void* output,
                                       const void* bias,
                                       MusaBroadcastParams params,
                                       bool has_bias,
                                       float beta,
                                       musaStream_t stream,
                                       int64_t inner_size) {
  if constexpr (std::is_same<T, double>::value) {
    GemmPostKernelDoubleNoActivation<kBroadcast><<<
        BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<double*>(output), reinterpret_cast<const double*>(bias),
        params, has_bias, beta, inner_size);
  } else {
    GemmPostKernelFloatNoActivation<T, kBroadcast><<<
        BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<T*>(output), reinterpret_cast<const T*>(bias), params,
        has_bias, beta, inner_size);
  }
  return musaGetLastError();
}

template <typename T, MusaGemmPostBroadcast kBroadcast>
musaError_t LaunchGemmPostBroadcast(void* output,
                                    const void* bias,
                                    MusaBroadcastParams params,
                                    bool has_bias,
                                    float beta,
                                    MusaUnaryOp activation,
                                    bool has_activation,
                                    float activation_alpha,
                                    musaStream_t stream,
                                    int64_t inner_size) {
  if (params.total_elements == 0 || (!has_bias && !has_activation)) {
    return musaSuccess;
  }
  if (!has_activation) {
    return LaunchGemmPostNoActivation<T, kBroadcast>(
        output, bias, params, has_bias, beta, stream, inner_size);
  }
  switch (activation) {
    case MusaUnaryOp::Relu:
      return LaunchGemmPostActivated<T, kBroadcast,
                                     GemmPostActivationKind::Relu>(
          output, bias, params, has_bias, beta, activation_alpha, stream,
          inner_size);
    case MusaUnaryOp::LeakyRelu:
      return LaunchGemmPostActivated<T, kBroadcast,
                                     GemmPostActivationKind::LeakyRelu>(
          output, bias, params, has_bias, beta, activation_alpha, stream,
          inner_size);
    case MusaUnaryOp::Sigmoid:
      return LaunchGemmPostActivated<T, kBroadcast,
                                     GemmPostActivationKind::Sigmoid>(
          output, bias, params, has_bias, beta, activation_alpha, stream,
          inner_size);
    default:
      return LaunchGemmPostGeneric<T, kBroadcast>(
          output, bias, params, has_bias, beta, activation, true,
          activation_alpha, stream, inner_size);
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
                                musaStream_t stream,
                                MusaGemmPostBroadcast broadcast,
                                int64_t inner_size) {
  switch (broadcast) {
    case MusaGemmPostBroadcast::Scalar:
      return LaunchGemmPostBroadcast<T, MusaGemmPostBroadcast::Scalar>(
          output, bias, params, has_bias, beta, activation, has_activation,
          activation_alpha, stream, inner_size);
    case MusaGemmPostBroadcast::LastDim:
      if (inner_size > 0) {
        return LaunchGemmPostBroadcast<T, MusaGemmPostBroadcast::LastDim>(
            output, bias, params, has_bias, beta, activation, has_activation,
            activation_alpha, stream, inner_size);
      }
      break;
    case MusaGemmPostBroadcast::Full:
      return LaunchGemmPostBroadcast<T, MusaGemmPostBroadcast::Full>(
          output, bias, params, has_bias, beta, activation, has_activation,
          activation_alpha, stream, inner_size);
    case MusaGemmPostBroadcast::Generic:
    default:
      break;
  }
  return LaunchGemmPostBroadcast<T, MusaGemmPostBroadcast::Generic>(
      output, bias, params, has_bias, beta, activation, has_activation,
      activation_alpha, stream, inner_size);
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
                                     MusaElementType elem_type, musaStream_t stream,
                                     MusaGemmPostBroadcast broadcast,
                                     int64_t broadcast_inner_size) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchGemmPostTyped<float>(output, bias, params, has_bias, beta,
                                        activation, has_activation,
                                        activation_alpha, stream, broadcast,
                                        broadcast_inner_size);
    case MusaElementType::Double:
      return LaunchGemmPostTyped<double>(output, bias, params, has_bias, beta,
                                         activation, has_activation,
                                         activation_alpha, stream, broadcast,
                                         broadcast_inner_size);
    case MusaElementType::Float16:
      return LaunchGemmPostTyped<__half>(output, bias, params, has_bias, beta,
                                         activation, has_activation,
                                         activation_alpha, stream, broadcast,
                                         broadcast_inner_size);
    case MusaElementType::BFloat16:
      return LaunchGemmPostTyped<__mt_bfloat16>(
          output, bias, params, has_bias, beta, activation, has_activation,
          activation_alpha, stream, broadcast, broadcast_inner_size);
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
