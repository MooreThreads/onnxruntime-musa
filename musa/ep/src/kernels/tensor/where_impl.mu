#include "tensor/where_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ void FastDivmod(
    int32_t value, const MusaFastDivmod& divisor, int32_t& quotient,
    int32_t& remainder) {
  const uint32_t high = __umulhi(divisor.multiplier,
                                 static_cast<uint32_t>(value));
  quotient = static_cast<int32_t>(
      (high + static_cast<uint32_t>(value)) >> divisor.shift);
  remainder = value - quotient * static_cast<int32_t>(divisor.divisor);
}

__device__ __forceinline__ void ResolveWhereInputIndices(
    int64_t index, const MusaWhereParams& params, int64_t& condition_index,
    int64_t& x_index, int64_t& y_index) {
  condition_index = params.condition_mode == 0 ? index : 0;
  x_index = params.x_mode == 0 ? index : 0;
  y_index = params.y_mode == 0 ? index : 0;
  int64_t remaining = index;
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    const int64_t coord = remaining / params.output_strides[dim];
    remaining -= coord * params.output_strides[dim];
    if (params.condition_mode == 2) {
      condition_index += coord * params.condition_strides[dim];
    }
    if (params.x_mode == 2) {
      x_index += coord * params.x_strides[dim];
    }
    if (params.y_mode == 2) {
      y_index += coord * params.y_strides[dim];
    }
  }
}

__device__ __forceinline__ void ResolveWhereInputIndicesFast(
    int32_t index, const MusaWhereParams& params, int64_t& condition_index,
    int64_t& x_index, int64_t& y_index) {
  condition_index = params.condition_mode == 0 ? index : 0;
  x_index = params.x_mode == 0 ? index : 0;
  y_index = params.y_mode == 0 ? index : 0;
  int32_t remaining = index;
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    int32_t coord;
    int32_t remainder;
    FastDivmod(remaining, params.output_divmod[dim], coord, remainder);
    remaining = remainder;
    if (params.condition_mode == 2) {
      condition_index += coord * params.condition_strides[dim];
    }
    if (params.x_mode == 2) {
      x_index += coord * params.x_strides[dim];
    }
    if (params.y_mode == 2) {
      y_index += coord * params.y_strides[dim];
    }
  }
}

template <typename T>
__device__ __forceinline__ void SelectElement(
    const void* x, const void* y, void* output, int64_t x_index,
    int64_t y_index, int64_t output_index, bool take_x) {
  reinterpret_cast<T*>(output)[output_index] =
      take_x ? reinterpret_cast<const T*>(x)[x_index]
             : reinterpret_cast<const T*>(y)[y_index];
}

template <typename T>
__global__ void WhereKernel(const uint8_t* condition,
                            const void* x,
                            const void* y,
                            void* output, MusaWhereParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.total_elements;
       output_index += total_threads) {
    int64_t condition_index;
    int64_t x_index;
    int64_t y_index;
    if (params.use_fast_divmod) {
      ResolveWhereInputIndicesFast(static_cast<int32_t>(output_index), params,
                                   condition_index, x_index, y_index);
    } else {
      ResolveWhereInputIndices(output_index, params, condition_index, x_index,
                               y_index);
    }
    SelectElement<T>(x, y, output, x_index, y_index, output_index,
                     condition[condition_index] != 0);
  }
}

template <typename T>
__global__ void WhereNoBroadcastKernel(const uint8_t* condition,
                                       const void* x, const void* y,
                                       void* output, int64_t total_elements) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < total_elements;
       index += total_threads) {
    SelectElement<T>(x, y, output, index, index, index,
                     condition[index] != 0);
  }
}

template <typename T>
musaError_t LaunchTypedWhere(const uint8_t* condition, const void* x,
                             const void* y, void* output,
                             const MusaWhereParams& params,
                             musaStream_t stream) {
  const int blocks = BlocksForCount(params.total_elements);
  if (params.condition_mode == 0 && params.x_mode == 0 &&
      params.y_mode == 0) {
    WhereNoBroadcastKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
        condition, x, y, output, params.total_elements);
  } else {
    WhereKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(condition, x, y,
                                                            output, params);
  }
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaWhereKernel(const uint8_t* condition,
                                  const void* x,
                                  const void* y,
                                  void* output,
                                  int32_t element_size,
                                  MusaWhereParams params,
                                  musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  switch (element_size) {
    case 1:
      return LaunchTypedWhere<uint8_t>(condition, x, y, output, params, stream);
    case 2:
      return LaunchTypedWhere<uint16_t>(condition, x, y, output, params, stream);
    case 4:
      return LaunchTypedWhere<uint32_t>(condition, x, y, output, params, stream);
    case 8:
      return LaunchTypedWhere<uint64_t>(condition, x, y, output, params, stream);
    default:
      return musaErrorInvalidValue;
  }
}
