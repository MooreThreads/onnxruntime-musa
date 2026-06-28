#include "math/cumsum_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T>
__device__ __forceinline__ T ZeroValue() {
  return static_cast<T>(0);
}

template <>
__device__ __forceinline__ __half ZeroValue<__half>() {
  return __float2half_rn(0.0f);
}

template <typename T>
__global__ void CumSumKernel(const T* input, T* output, int64_t output_size,
                             int64_t axis_dim, int64_t axis_stride,
                             bool exclusive, bool reverse) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < output_size; index += total_threads) {
    const int64_t axis_coord = (index / axis_stride) % axis_dim;

    int64_t start = 0;
    int64_t end = 0;
    if (!reverse && !exclusive) {
      start = 0;
      end = axis_coord;
    } else if (reverse && !exclusive) {
      start = axis_coord;
      end = axis_dim - 1;
    } else if (!reverse && exclusive) {
      start = 0;
      end = axis_coord - 1;
    } else {
      start = axis_coord + 1;
      end = axis_dim - 1;
    }

    if (end < start) {
      output[index] = ZeroValue<T>();
      continue;
    }

    T sum = ZeroValue<T>();
    int64_t data_index = index + (start - axis_coord) * axis_stride;
    for (int64_t i = start; i <= end; ++i) {
      sum += input[data_index];
      data_index += axis_stride;
    }
    output[index] = sum;
  }
}

template <typename T>
musaError_t LaunchTypedCumSum(const void* input, void* output,
                              int64_t output_size, int64_t axis_dim,
                              int64_t axis_stride, bool exclusive,
                              bool reverse, musaStream_t stream) {
  if (output_size == 0) {
    return musaSuccess;
  }
  CumSumKernel<T><<<BlocksForCount(output_size), kThreadsPerBlock, 0, stream>>>(
      static_cast<const T*>(input), static_cast<T*>(output), output_size,
      axis_dim, axis_stride, exclusive, reverse);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaCumSumKernel(const void* input, void* output,
                                   int32_t elem_type, int64_t output_size,
                                   int64_t axis_dim,
                                   int64_t axis_stride, bool exclusive,
                                   bool reverse, musaStream_t stream) {
  constexpr int32_t kFloat = 1;
  constexpr int32_t kInt32 = 6;
  constexpr int32_t kInt64 = 7;
  constexpr int32_t kFloat16 = 10;
  constexpr int32_t kDouble = 11;
  constexpr int32_t kUint32 = 12;
  constexpr int32_t kUint64 = 13;
  switch (elem_type) {
    case kInt32:
      return LaunchTypedCumSum<int32_t>(input, output, output_size, axis_dim,
                                        axis_stride, exclusive, reverse, stream);
    case kInt64:
      return LaunchTypedCumSum<int64_t>(input, output, output_size, axis_dim,
                                        axis_stride, exclusive, reverse, stream);
    case kUint32:
      return LaunchTypedCumSum<uint32_t>(input, output, output_size, axis_dim,
                                         axis_stride, exclusive, reverse, stream);
    case kUint64:
      return LaunchTypedCumSum<uint64_t>(input, output, output_size, axis_dim,
                                         axis_stride, exclusive, reverse, stream);
    case kFloat:
      return LaunchTypedCumSum<float>(input, output, output_size, axis_dim,
                                      axis_stride, exclusive, reverse, stream);
    case kDouble:
      return LaunchTypedCumSum<double>(input, output, output_size, axis_dim,
                                       axis_stride, exclusive, reverse, stream);
    case kFloat16:
      return LaunchTypedCumSum<__half>(input, output, output_size, axis_dim,
                                       axis_stride, exclusive, reverse, stream);
    default:
      return musaErrorInvalidValue;
  }
}
