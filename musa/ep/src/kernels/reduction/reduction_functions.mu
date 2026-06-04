#include "reduction/reduction_functions.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename AccT, typename T>
__device__ __forceinline__ AccT ReduceToAccum(T value) {
  return static_cast<AccT>(value);
}

template <typename AccT>
__device__ __forceinline__ AccT ReduceToAccum(__half value) {
  return static_cast<AccT>(__half2float(value));
}

template <typename AccT>
__device__ __forceinline__ AccT ReduceToAccum(__mt_bfloat16 value) {
  return static_cast<AccT>(__bfloat162float(value));
}

template <typename T, typename AccT>
__device__ __forceinline__ T ReduceFromAccum(AccT value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half ReduceFromAccum<__half, float>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __mt_bfloat16
ReduceFromAccum<__mt_bfloat16, float>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename AccT>
__device__ __forceinline__ AccT ReduceLowestValue() {
  return static_cast<AccT>(0);
}

template <>
__device__ __forceinline__ float ReduceLowestValue<float>() {
  return -INFINITY;
}

template <>
__device__ __forceinline__ double ReduceLowestValue<double>() {
  return -INFINITY;
}

template <>
__device__ __forceinline__ int32_t ReduceLowestValue<int32_t>() {
  return INT32_MIN;
}

template <>
__device__ __forceinline__ int64_t ReduceLowestValue<int64_t>() {
  return INT64_MIN;
}

template <typename AccT>
__device__ __forceinline__ AccT ReduceInitValue(MusaReduceOp op) {
  if (op == MusaReduceOp::Prod) {
    return static_cast<AccT>(1);
  }
  if (op == MusaReduceOp::Max) {
    return ReduceLowestValue<AccT>();
  }
  return static_cast<AccT>(0);
}

template <typename AccT>
__device__ __forceinline__ AccT ReduceUpdateValue(AccT acc, AccT value,
                                                  MusaReduceOp op) {
  if (op == MusaReduceOp::Prod) {
    return acc * value;
  }
  if (op == MusaReduceOp::SumSquare) {
    return acc + value * value;
  }
  if (op == MusaReduceOp::Max) {
    return acc > value ? acc : value;
  }
  return acc + value;
}

template <typename AccT>
__device__ __forceinline__ AccT ReduceCombineValue(AccT lhs, AccT rhs,
                                                   MusaReduceOp op) {
  if (op == MusaReduceOp::Prod) {
    return lhs * rhs;
  }
  if (op == MusaReduceOp::Max) {
    return lhs > rhs ? lhs : rhs;
  }
  return lhs + rhs;
}

__device__ __forceinline__ int64_t ReduceInputBase(int64_t output_index,
                                                   MusaReduceParams params) {
  int64_t remaining = output_index;
  int64_t input_base = 0;
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    if (params.reduce_axes[dim] != 0) {
      continue;
    }
    const int64_t stride = params.output_strides[dim];
    const int64_t coord = stride == 0 ? 0 : remaining / stride;
    if (stride != 0) {
      remaining -= coord * stride;
    }
    input_base += coord * params.input_strides[dim];
  }
  return input_base;
}

__device__ __forceinline__ int64_t ReduceInputOffset(
    int64_t reduction_index,
    MusaReduceParams params) {
  int64_t remaining = reduction_index;
  int64_t input_offset = 0;
  for (int32_t dim = params.rank - 1; dim >= 0; --dim) {
    if (params.reduce_axes[dim] == 0) {
      continue;
    }
    const int64_t size = params.input_dims[dim];
    const int64_t coord = size == 0 ? 0 : remaining % size;
    remaining = size == 0 ? 0 : remaining / size;
    input_offset += coord * params.input_strides[dim];
  }
  return input_offset;
}

template <typename T, typename AccT>
__global__ void ReduceKernel(const T* input,
                             T* output,
                             MusaReduceParams params,
                             MusaReduceOp op) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.output_elements;
       output_index += total_threads) {
    const int64_t input_base = ReduceInputBase(output_index, params);

    AccT acc = ReduceInitValue<AccT>(op);
    for (int64_t r = 0; r < params.reduction_elements; ++r) {
      const int64_t input_index = input_base + ReduceInputOffset(r, params);
      acc =
          ReduceUpdateValue(acc, ReduceToAccum<AccT>(input[input_index]), op);
    }
    if (op == MusaReduceOp::Mean) {
      acc /= static_cast<AccT>(params.reduction_elements);
    }
    output[output_index] = ReduceFromAccum<T, AccT>(acc);
  }
}

template <typename T, typename AccT>
__global__ void ReduceBlockKernel(const T* input,
                                  T* output,
                                  MusaReduceParams params,
                                  MusaReduceOp op) {
  const int64_t output_index = static_cast<int64_t>(blockIdx.x);
  if (output_index >= params.output_elements) {
    return;
  }

  const int64_t input_base = ReduceInputBase(output_index, params);

  AccT acc = ReduceInitValue<AccT>(op);
  for (int64_t r = threadIdx.x; r < params.reduction_elements;
       r += blockDim.x) {
    const int64_t input_index = input_base + ReduceInputOffset(r, params);
    acc = ReduceUpdateValue(acc, ReduceToAccum<AccT>(input[input_index]), op);
  }

  __shared__ AccT shared[kThreadsPerBlock];
  shared[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          ReduceCombineValue(shared[threadIdx.x], shared[threadIdx.x + stride],
                             op);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    AccT value = shared[0];
    if (op == MusaReduceOp::Mean) {
      value /= static_cast<AccT>(params.reduction_elements);
    }
    output[output_index] = ReduceFromAccum<T, AccT>(value);
  }
}

}  // namespace

template <typename T, typename AccT>
musaError_t LaunchMusaReduceTyped(const void* input,
                                  void* output,
                                  MusaReduceParams params,
                                  MusaReduceOp op,
                                  musaStream_t stream) {
  if (params.output_elements == 0) {
    return musaSuccess;
  }
  if (params.reduction_elements >= 1024 ||
      (params.reduction_elements >= 32 && params.output_elements <= 4096)) {
    ReduceBlockKernel<T, AccT>
        <<<static_cast<int>(params.output_elements), kThreadsPerBlock, 0,
           stream>>>(reinterpret_cast<const T*>(input),
                     reinterpret_cast<T*>(output), params, op);
    return musaGetLastError();
  }
  ReduceKernel<T, AccT>
      <<<BlocksForCount(params.output_elements), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output),
          params, op);
  return musaGetLastError();
}

musaError_t LaunchMusaReduceKernel(const void* input,
                                   void* output,
                                   MusaReduceParams params,
                                   MusaReduceOp op,
                                   MusaElementType elem_type,
                                   musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchMusaReduceTyped<float, float>(input, output, params, op,
                                                 stream);
    case MusaElementType::Double:
      return LaunchMusaReduceTyped<double, double>(input, output, params, op,
                                                   stream);
    case MusaElementType::Int32:
      if (op == MusaReduceOp::Mean) {
        return LaunchMusaReduceTyped<int32_t, int64_t>(input, output, params,
                                                       op, stream);
      }
      return LaunchMusaReduceTyped<int32_t, int32_t>(input, output, params, op,
                                                     stream);
    case MusaElementType::Int64:
      if (op == MusaReduceOp::Mean) return musaErrorNotSupported;
      return LaunchMusaReduceTyped<int64_t, int64_t>(input, output, params, op,
                                                     stream);
    case MusaElementType::Float16:
      return LaunchMusaReduceTyped<__half, float>(input, output, params, op,
                                                  stream);
    case MusaElementType::BFloat16:
      return LaunchMusaReduceTyped<__mt_bfloat16, float>(
          input, output, params, op, stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaReduceFloatKernel(const float* input,
                                        float* output,
                                        MusaReduceParams params,
                                        MusaReduceOp op,
                                        musaStream_t stream) {
  return LaunchMusaReduceKernel(input, output, params, op,
                                MusaElementType::Float, stream);
}

musaError_t LaunchMusaReduceInt32Kernel(const int32_t* input,
                                        int32_t* output,
                                        MusaReduceParams params,
                                        MusaReduceOp op,
                                        musaStream_t stream) {
  return LaunchMusaReduceKernel(input, output, params, op,
                                MusaElementType::Int32, stream);
}

musaError_t LaunchMusaReduceInt64Kernel(const int64_t* input,
                                        int64_t* output,
                                        MusaReduceParams params,
                                        MusaReduceOp op,
                                        musaStream_t stream) {
  return LaunchMusaReduceKernel(input, output, params, op,
                                MusaElementType::Int64, stream);
}
