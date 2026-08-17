#include "math/binary_elementwise_ops_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <MusaBinaryOp Op>
struct BinaryOp {
  template <typename T>
  __device__ __forceinline__ static T Apply(T lhs, T rhs) {
    return lhs;
  }
};

#define DEFINE_BINARY_OP(OP, EXPRESSION)                                      \
  template <>                                                                \
  struct BinaryOp<MusaBinaryOp::OP> {                                        \
    template <typename T>                                                    \
    __device__ __forceinline__ static T Apply(T lhs, T rhs) {                 \
      return EXPRESSION;                                                      \
    }                                                                         \
  }

DEFINE_BINARY_OP(Add, lhs + rhs);
DEFINE_BINARY_OP(Sub, lhs - rhs);
DEFINE_BINARY_OP(Mul, lhs * rhs);
DEFINE_BINARY_OP(Div, lhs / rhs);
DEFINE_BINARY_OP(Max, lhs > rhs ? lhs : rhs);
DEFINE_BINARY_OP(Min, lhs < rhs ? lhs : rhs);

template <>
struct BinaryOp<MusaBinaryOp::Pow> {
  template <typename T>
  __device__ __forceinline__ static T Apply(T lhs, T rhs) {
    return static_cast<T>(
        pow(static_cast<double>(lhs), static_cast<double>(rhs)));
  }
};

#undef DEFINE_BINARY_OP

template <MusaBinaryOp Op>
__device__ __forceinline__ float BinaryFloatValue(float lhs, float rhs) {
  return BinaryOp<Op>::Apply(lhs, rhs);
}

template <>
__device__ __forceinline__ float BinaryFloatValue<MusaBinaryOp::Pow>(
    float lhs, float rhs) {
  return powf(lhs, rhs);
}

template <typename T>
__device__ __forceinline__ uint8_t CompareValue(T lhs, T rhs, MusaCompareOp op) {
  switch (op) {
    case MusaCompareOp::Equal:
      return static_cast<uint8_t>(lhs == rhs);
    case MusaCompareOp::Greater:
      return static_cast<uint8_t>(lhs > rhs);
    case MusaCompareOp::Less:
      return static_cast<uint8_t>(lhs < rhs);
    case MusaCompareOp::GreaterOrEqual:
      return static_cast<uint8_t>(lhs >= rhs);
    case MusaCompareOp::LessOrEqual:
      return static_cast<uint8_t>(lhs <= rhs);
  }
  return 0;
}

template <typename T, MusaBinaryOp Op>
__global__ void BinaryKernel(const T* lhs,
                             const T* rhs,
                             T* output,
                             MusaBroadcastParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    output[index] = BinaryOp<Op>::Apply(lhs[lhs_index], rhs[rhs_index]);
  }
}

template <typename T, MusaBinaryOp Op>
__global__ void BinaryContiguousKernel(const T* lhs,
                                       const T* rhs,
                                       T* output,
                                       int64_t total_elements) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < total_elements;
       index += total_threads) {
    output[index] = BinaryOp<Op>::Apply(lhs[index], rhs[index]);
  }
}

template <typename T, MusaBinaryOp Op, bool LhsScalar>
__global__ void BinaryScalarKernel(const T* lhs,
                                   const T* rhs,
                                   T* output,
                                   int64_t total_elements) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const T scalar = LhsScalar ? lhs[0] : rhs[0];
  for (int64_t index = thread_id; index < total_elements;
       index += total_threads) {
    const T lhs_value = LhsScalar ? scalar : lhs[index];
    const T rhs_value = LhsScalar ? rhs[index] : scalar;
    output[index] = BinaryOp<Op>::Apply(lhs_value, rhs_value);
  }
}

template <MusaBinaryOp Op>
__global__ void BinaryFloatKernel(const float* lhs,
                                  const float* rhs,
                                  float* output,
                                  MusaBroadcastParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    output[index] = BinaryFloatValue<Op>(lhs[lhs_index], rhs[rhs_index]);
  }
}

template <typename T, MusaBinaryOp Op>
__global__ void BinaryFloatLikeKernel(const T* lhs,
                                      const T* rhs,
                                      T* output,
                                      MusaBroadcastParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    const float lhs_value = MusaScalarToFloat(lhs[lhs_index]);
    const float rhs_value = MusaScalarToFloat(rhs[rhs_index]);
    output[index] = MusaScalarFromFloat<T>(
        BinaryFloatValue<Op>(lhs_value, rhs_value));
  }
}

template <typename T, MusaBinaryOp Op>
__global__ void BinaryFloatLikeContiguousKernel(const T* lhs,
                                                const T* rhs,
                                                T* output,
                                                int64_t total_elements) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < total_elements;
       index += total_threads) {
    const float lhs_value = MusaScalarToFloat(lhs[index]);
    const float rhs_value = MusaScalarToFloat(rhs[index]);
    output[index] = MusaScalarFromFloat<T>(
        BinaryFloatValue<Op>(lhs_value, rhs_value));
  }
}

template <typename T, MusaBinaryOp Op, bool LhsScalar>
__global__ void BinaryFloatLikeScalarKernel(const T* lhs,
                                             const T* rhs,
                                             T* output,
                                             int64_t total_elements) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const float scalar = MusaScalarToFloat<T>(LhsScalar ? lhs[0] : rhs[0]);
  for (int64_t index = thread_id; index < total_elements;
       index += total_threads) {
    const float lhs_value = LhsScalar ? scalar : MusaScalarToFloat(lhs[index]);
    const float rhs_value = LhsScalar ? MusaScalarToFloat(rhs[index]) : scalar;
    output[index] = MusaScalarFromFloat<T>(
        BinaryFloatValue<Op>(lhs_value, rhs_value));
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

inline bool BinaryParamsAreContiguous(const MusaBroadcastParams& params) {
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    if (params.lhs_strides[dim] != params.output_strides[dim] ||
        params.rhs_strides[dim] != params.output_strides[dim]) {
      return false;
    }
  }
  return true;
}

inline bool BinaryParamsLhsScalar(const MusaBroadcastParams& params) {
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    if (params.lhs_strides[dim] != 0) return false;
  }
  return true;
}

inline bool BinaryParamsRhsScalar(const MusaBroadcastParams& params) {
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    if (params.rhs_strides[dim] != 0) return false;
  }
  return true;
}

template <typename T, MusaBinaryOp Op>
musaError_t LaunchBinaryTyped(const void* lhs,
                              const void* rhs,
                              void* output,
                              MusaBroadcastParams params,
                              musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  const int blocks = BlocksForCount(params.total_elements);
  if (BinaryParamsAreContiguous(params)) {
    BinaryContiguousKernel<T, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const T*>(lhs), reinterpret_cast<const T*>(rhs),
        reinterpret_cast<T*>(output), params.total_elements);
  } else if (BinaryParamsLhsScalar(params)) {
    BinaryScalarKernel<T, Op, true><<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const T*>(lhs), reinterpret_cast<const T*>(rhs),
        reinterpret_cast<T*>(output), params.total_elements);
  } else if (BinaryParamsRhsScalar(params)) {
    BinaryScalarKernel<T, Op, false><<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const T*>(lhs), reinterpret_cast<const T*>(rhs),
        reinterpret_cast<T*>(output), params.total_elements);
  } else {
    BinaryKernel<T, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const T*>(lhs), reinterpret_cast<const T*>(rhs),
        reinterpret_cast<T*>(output), params);
  }
  return musaGetLastError();
}

template <typename T, MusaBinaryOp Op>
musaError_t LaunchBinaryFloatLikeTyped(const void* lhs,
                                       const void* rhs,
                                       void* output,
                                       MusaBroadcastParams params,
                                       musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  const int blocks = BlocksForCount(params.total_elements);
  if (BinaryParamsAreContiguous(params)) {
    BinaryFloatLikeContiguousKernel<T, Op><<<blocks, kThreadsPerBlock, 0,
                                            stream>>>(
        reinterpret_cast<const T*>(lhs), reinterpret_cast<const T*>(rhs),
        reinterpret_cast<T*>(output), params.total_elements);
  } else if (BinaryParamsLhsScalar(params)) {
    BinaryFloatLikeScalarKernel<T, Op, true><<<blocks, kThreadsPerBlock, 0,
                                               stream>>>(
        reinterpret_cast<const T*>(lhs), reinterpret_cast<const T*>(rhs),
        reinterpret_cast<T*>(output), params.total_elements);
  } else if (BinaryParamsRhsScalar(params)) {
    BinaryFloatLikeScalarKernel<T, Op, false><<<blocks, kThreadsPerBlock, 0,
                                                stream>>>(
        reinterpret_cast<const T*>(lhs), reinterpret_cast<const T*>(rhs),
        reinterpret_cast<T*>(output), params.total_elements);
  } else {
    BinaryFloatLikeKernel<T, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const T*>(lhs), reinterpret_cast<const T*>(rhs),
        reinterpret_cast<T*>(output), params);
  }
  return musaGetLastError();
}

musaError_t LaunchMusaBinaryKernelForOp(const void* lhs,
                                        const void* rhs,
                                        void* output,
                                        MusaBroadcastParams params,
                                        MusaBinaryOp op,
                                        MusaElementType elem_type,
                                        musaStream_t stream);

musaError_t LaunchMusaBinaryKernel(const void* lhs,
                                   const void* rhs,
                                   void* output,
                                   MusaBroadcastParams params,
                                   MusaBinaryOp op,
                                   MusaElementType elem_type,
                                   musaStream_t stream) {
  return LaunchMusaBinaryKernelForOp(lhs, rhs, output, params, op, elem_type,
                                     stream);
}

template <MusaBinaryOp Op>
musaError_t LaunchMusaBinaryKernelTyped(const void* lhs,
                                        const void* rhs,
                                        void* output,
                                        MusaBroadcastParams params,
                                        MusaElementType elem_type,
                                        musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchBinaryTyped<float, Op>(lhs, rhs, output, params, stream);
    case MusaElementType::Double:
      return LaunchBinaryTyped<double, Op>(lhs, rhs, output, params, stream);
    case MusaElementType::Uint8:
      return LaunchBinaryTyped<uint8_t, Op>(lhs, rhs, output, params, stream);
    case MusaElementType::Uint16:
      return LaunchBinaryTyped<uint16_t, Op>(lhs, rhs, output, params, stream);
    case MusaElementType::Uint32:
      return LaunchBinaryTyped<uint32_t, Op>(lhs, rhs, output, params, stream);
    case MusaElementType::Uint64:
      return LaunchBinaryTyped<uint64_t, Op>(lhs, rhs, output, params, stream);
    case MusaElementType::Int8:
      return LaunchBinaryTyped<int8_t, Op>(lhs, rhs, output, params, stream);
    case MusaElementType::Int16:
      return LaunchBinaryTyped<int16_t, Op>(lhs, rhs, output, params, stream);
    case MusaElementType::Int32:
      return LaunchBinaryTyped<int32_t, Op>(lhs, rhs, output, params, stream);
    case MusaElementType::Int64:
      return LaunchBinaryTyped<int64_t, Op>(lhs, rhs, output, params, stream);
    case MusaElementType::Float16:
      return LaunchBinaryFloatLikeTyped<__half, Op>(lhs, rhs, output, params,
                                                    stream);
    case MusaElementType::BFloat16:
      return LaunchBinaryFloatLikeTyped<__mt_bfloat16, Op>(
          lhs, rhs, output, params, stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaBinaryKernelForOp(const void* lhs,
                                        const void* rhs,
                                        void* output,
                                        MusaBroadcastParams params,
                                        MusaBinaryOp op,
                                        MusaElementType elem_type,
                                        musaStream_t stream) {
  switch (op) {
    case MusaBinaryOp::Add:
      return LaunchMusaBinaryKernelTyped<MusaBinaryOp::Add>(
          lhs, rhs, output, params, elem_type, stream);
    case MusaBinaryOp::Sub:
      return LaunchMusaBinaryKernelTyped<MusaBinaryOp::Sub>(
          lhs, rhs, output, params, elem_type, stream);
    case MusaBinaryOp::Mul:
      return LaunchMusaBinaryKernelTyped<MusaBinaryOp::Mul>(
          lhs, rhs, output, params, elem_type, stream);
    case MusaBinaryOp::Div:
      return LaunchMusaBinaryKernelTyped<MusaBinaryOp::Div>(
          lhs, rhs, output, params, elem_type, stream);
    case MusaBinaryOp::Pow:
      return LaunchMusaBinaryKernelTyped<MusaBinaryOp::Pow>(
          lhs, rhs, output, params, elem_type, stream);
    case MusaBinaryOp::Max:
      return LaunchMusaBinaryKernelTyped<MusaBinaryOp::Max>(
          lhs, rhs, output, params, elem_type, stream);
    case MusaBinaryOp::Min:
      return LaunchMusaBinaryKernelTyped<MusaBinaryOp::Min>(
          lhs, rhs, output, params, elem_type, stream);
  }
  return musaErrorNotSupported;
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
