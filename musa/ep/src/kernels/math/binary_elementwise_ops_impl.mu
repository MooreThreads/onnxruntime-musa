#include "math/binary_elementwise_ops_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ float BinaryValue(float lhs, float rhs, MusaBinaryOp op) {
  switch (op) {
    case MusaBinaryOp::Add:
      return lhs + rhs;
    case MusaBinaryOp::Sub:
      return lhs - rhs;
    case MusaBinaryOp::Mul:
      return lhs * rhs;
    case MusaBinaryOp::Div:
      return lhs / rhs;
    case MusaBinaryOp::Pow:
      return powf(lhs, rhs);
    case MusaBinaryOp::Max:
      return lhs > rhs ? lhs : rhs;
    case MusaBinaryOp::Min:
      return lhs < rhs ? lhs : rhs;
  }
  return lhs;
}

template <typename T>
__device__ __forceinline__ T BinaryValueTyped(T lhs, T rhs, MusaBinaryOp op) {
  switch (op) {
    case MusaBinaryOp::Add:
      return static_cast<T>(lhs + rhs);
    case MusaBinaryOp::Sub:
      return static_cast<T>(lhs - rhs);
    case MusaBinaryOp::Mul:
      return static_cast<T>(lhs * rhs);
    case MusaBinaryOp::Div:
      return static_cast<T>(lhs / rhs);
    case MusaBinaryOp::Pow:
      return static_cast<T>(
          pow(static_cast<double>(lhs), static_cast<double>(rhs)));
    case MusaBinaryOp::Max:
      return lhs > rhs ? lhs : rhs;
    case MusaBinaryOp::Min:
      return lhs < rhs ? lhs : rhs;
  }
  return lhs;
}

template <typename T>
__device__ __forceinline__ uint8_t CompareValue(T lhs, T rhs, MusaCompareOp op) {
  switch (op) {
    case MusaCompareOp::Equal:
      return static_cast<uint8_t>(lhs == rhs);
    case MusaCompareOp::Greater:
      return static_cast<uint8_t>(lhs > rhs);
  }
  return 0;
}

template <typename T>
__global__ void BinaryKernel(const T* lhs,
                             const T* rhs,
                             T* output,
                             MusaBroadcastParams params,
                             MusaBinaryOp op) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    output[index] = BinaryValueTyped(lhs[lhs_index], rhs[rhs_index], op);
  }
}

__global__ void BinaryFloatKernel(const float* lhs,
                                  const float* rhs,
                                  float* output,
                                  MusaBroadcastParams params,
                                  MusaBinaryOp op) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    output[index] = BinaryValue(lhs[lhs_index], rhs[rhs_index], op);
  }
}

template <typename T>
__global__ void BinaryFloatLikeKernel(const T* lhs,
                                      const T* rhs,
                                      T* output,
                                      MusaBroadcastParams params,
                                      MusaBinaryOp op) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    const float lhs_value = MusaScalarToFloat(lhs[lhs_index]);
    const float rhs_value = MusaScalarToFloat(rhs[rhs_index]);
    output[index] = MusaScalarFromFloat<T>(BinaryValue(lhs_value, rhs_value, op));
  }
}

template <typename T>
__global__ void CompareKernel(const T* lhs,
                              const T* rhs,
                              uint8_t* output,
                              MusaBroadcastParams params,
                              MusaCompareOp op) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    output[index] = CompareValue(lhs[lhs_index], rhs[rhs_index], op);
  }
}

template <typename T>
__global__ void CompareFloatLikeKernel(const T* lhs,
                                       const T* rhs,
                                       uint8_t* output,
                                       MusaBroadcastParams params,
                                       MusaCompareOp op) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    const float lhs_value = MusaScalarToFloat(lhs[lhs_index]);
    const float rhs_value = MusaScalarToFloat(rhs[rhs_index]);
    output[index] = CompareValue(lhs_value, rhs_value, op);
  }
}

template <typename T, typename TExponent>
__global__ void PowMixedKernel(const T* lhs,
                               const TExponent* rhs,
                               T* output,
                               MusaBroadcastParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    const double lhs_value = MusaScalarToDouble(lhs[lhs_index]);
    const double rhs_value = MusaScalarToDouble(rhs[rhs_index]);
    output[index] = MusaScalarFromDouble<T>(pow(lhs_value, rhs_value));
  }
}

}  // namespace

template <typename T>
musaError_t LaunchBinaryTyped(const void* lhs,
                              const void* rhs,
                              void* output,
                              MusaBroadcastParams params,
                              MusaBinaryOp op,
                              musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  BinaryKernel<T><<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(lhs), reinterpret_cast<const T*>(rhs),
      reinterpret_cast<T*>(output), params, op);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchBinaryFloatLikeTyped(const void* lhs,
                                       const void* rhs,
                                       void* output,
                                       MusaBroadcastParams params,
                                       MusaBinaryOp op,
                                       musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  BinaryFloatLikeKernel<T><<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(lhs), reinterpret_cast<const T*>(rhs),
      reinterpret_cast<T*>(output), params, op);
  return musaGetLastError();
}

musaError_t LaunchMusaBinaryKernel(const void* lhs,
                                   const void* rhs,
                                   void* output,
                                   MusaBroadcastParams params,
                                   MusaBinaryOp op,
                                   MusaElementType elem_type,
                                   musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchBinaryTyped<float>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Double:
      return LaunchBinaryTyped<double>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Uint8:
      return LaunchBinaryTyped<uint8_t>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Uint16:
      return LaunchBinaryTyped<uint16_t>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Uint32:
      return LaunchBinaryTyped<uint32_t>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Uint64:
      return LaunchBinaryTyped<uint64_t>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Int8:
      return LaunchBinaryTyped<int8_t>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Int16:
      return LaunchBinaryTyped<int16_t>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Int32:
      return LaunchBinaryTyped<int32_t>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Int64:
      return LaunchBinaryTyped<int64_t>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Float16:
      return LaunchBinaryFloatLikeTyped<__half>(lhs, rhs, output, params, op,
                                                stream);
    case MusaElementType::BFloat16:
      return LaunchBinaryFloatLikeTyped<__mt_bfloat16>(
          lhs, rhs, output, params, op, stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaBinaryFloatKernel(const float* lhs,
                                        const float* rhs,
                                        float* output,
                                        MusaBroadcastParams params,
                                        MusaBinaryOp op,
                                        musaStream_t stream) {
  return LaunchMusaBinaryKernel(lhs, rhs, output, params, op,
                                MusaElementType::Float, stream);
}

template <typename T, typename TExponent>
musaError_t LaunchPowMixedTyped(const void* lhs,
                                const void* rhs,
                                void* output,
                                MusaBroadcastParams params,
                                musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  PowMixedKernel<T, TExponent><<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(lhs), reinterpret_cast<const TExponent*>(rhs),
      reinterpret_cast<T*>(output), params);
  return musaGetLastError();
}

template <typename T>
musaError_t DispatchPowExponentType(const void* lhs,
                                    const void* rhs,
                                    void* output,
                                    MusaBroadcastParams params,
                                    MusaElementType rhs_elem_type,
                                    musaStream_t stream) {
  switch (rhs_elem_type) {
    case MusaElementType::Int32:
      return LaunchPowMixedTyped<T, int32_t>(lhs, rhs, output, params, stream);
    case MusaElementType::Int64:
      return LaunchPowMixedTyped<T, int64_t>(lhs, rhs, output, params, stream);
    case MusaElementType::Float16:
      return LaunchPowMixedTyped<T, __half>(lhs, rhs, output, params, stream);
    case MusaElementType::Float:
      return LaunchPowMixedTyped<T, float>(lhs, rhs, output, params, stream);
    case MusaElementType::Double:
      return LaunchPowMixedTyped<T, double>(lhs, rhs, output, params, stream);
    case MusaElementType::BFloat16:
      return LaunchPowMixedTyped<T, __mt_bfloat16>(lhs, rhs, output, params,
                                                   stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaPowKernel(const void* lhs,
                                const void* rhs,
                                void* output,
                                MusaBroadcastParams params,
                                MusaElementType lhs_elem_type,
                                MusaElementType rhs_elem_type,
                                musaStream_t stream) {
  switch (lhs_elem_type) {
    case MusaElementType::Int32:
      return DispatchPowExponentType<int32_t>(lhs, rhs, output, params,
                                              rhs_elem_type, stream);
    case MusaElementType::Int64:
      return DispatchPowExponentType<int64_t>(lhs, rhs, output, params,
                                              rhs_elem_type, stream);
    case MusaElementType::Float16:
      return DispatchPowExponentType<__half>(lhs, rhs, output, params,
                                             rhs_elem_type, stream);
    case MusaElementType::Float:
      return DispatchPowExponentType<float>(lhs, rhs, output, params,
                                            rhs_elem_type, stream);
    case MusaElementType::Double:
      return DispatchPowExponentType<double>(lhs, rhs, output, params,
                                             rhs_elem_type, stream);
    case MusaElementType::BFloat16:
      return DispatchPowExponentType<__mt_bfloat16>(lhs, rhs, output, params,
                                                    rhs_elem_type, stream);
    default:
      return musaErrorNotSupported;
  }
}

template <typename T>
musaError_t LaunchCompareTyped(const void* lhs,
                               const void* rhs,
                               uint8_t* output,
                               MusaBroadcastParams params,
                               MusaCompareOp op,
                               musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  CompareKernel<T><<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(lhs), reinterpret_cast<const T*>(rhs), output,
      params, op);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchCompareFloatLikeTyped(const void* lhs,
                                        const void* rhs,
                                        uint8_t* output,
                                        MusaBroadcastParams params,
                                        MusaCompareOp op,
                                        musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  CompareFloatLikeKernel<T><<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(lhs), reinterpret_cast<const T*>(rhs), output,
      params, op);
  return musaGetLastError();
}

musaError_t LaunchMusaCompareKernel(const void* lhs,
                                    const void* rhs,
                                    uint8_t* output,
                                    MusaBroadcastParams params,
                                    MusaCompareOp op,
                                    MusaElementType elem_type,
                                    musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchCompareTyped<float>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Double:
      return LaunchCompareTyped<double>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Uint32:
      return LaunchCompareTyped<uint32_t>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Uint64:
      return LaunchCompareTyped<uint64_t>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Int32:
      return LaunchCompareTyped<int32_t>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Int64:
      return LaunchCompareTyped<int64_t>(lhs, rhs, output, params, op, stream);
    case MusaElementType::Float16:
      return LaunchCompareFloatLikeTyped<__half>(lhs, rhs, output, params, op,
                                                 stream);
    case MusaElementType::BFloat16:
      return LaunchCompareFloatLikeTyped<__mt_bfloat16>(
          lhs, rhs, output, params, op, stream);
    case MusaElementType::Bool:
      if (op != MusaCompareOp::Equal) {
        return musaErrorNotSupported;
      }
      return LaunchCompareTyped<uint8_t>(lhs, rhs, output, params, op, stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaCompareFloatKernel(const float* lhs,
                                         const float* rhs,
                                         uint8_t* output,
                                         MusaBroadcastParams params,
                                         MusaCompareOp op,
                                         musaStream_t stream) {
  return LaunchMusaCompareKernel(lhs, rhs, output, params, op,
                                 MusaElementType::Float, stream);
}

musaError_t LaunchMusaCompareInt32Kernel(const int32_t* lhs,
                                         const int32_t* rhs,
                                         uint8_t* output,
                                         MusaBroadcastParams params,
                                         MusaCompareOp op,
                                         musaStream_t stream) {
  return LaunchMusaCompareKernel(lhs, rhs, output, params, op,
                                 MusaElementType::Int32, stream);
}

musaError_t LaunchMusaCompareInt64Kernel(const int64_t* lhs,
                                         const int64_t* rhs,
                                         uint8_t* output,
                                         MusaBroadcastParams params,
                                         MusaCompareOp op,
                                         musaStream_t stream) {
  return LaunchMusaCompareKernel(lhs, rhs, output, params, op,
                                 MusaElementType::Int64, stream);
}
