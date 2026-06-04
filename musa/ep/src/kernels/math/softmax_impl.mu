#include "math/softmax_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename AccT, typename T>
__device__ __forceinline__ AccT SoftmaxToAccum(T value) {
  return static_cast<AccT>(value);
}

template <typename AccT>
__device__ __forceinline__ AccT SoftmaxToAccum(__half value) {
  return static_cast<AccT>(__half2float(value));
}

template <typename AccT>
__device__ __forceinline__ AccT SoftmaxToAccum(__mt_bfloat16 value) {
  return static_cast<AccT>(__bfloat162float(value));
}

template <typename T, typename AccT>
__device__ __forceinline__ T SoftmaxFromAccum(AccT value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half SoftmaxFromAccum<__half, float>(
    float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __mt_bfloat16
SoftmaxFromAccum<__mt_bfloat16, float>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename AccT>
__device__ __forceinline__ AccT SoftmaxExp(AccT value) {
  return exp(value);
}

template <>
__device__ __forceinline__ float SoftmaxExp<float>(float value) {
  return expf(value);
}

template <typename T, typename AccT>
__global__ void SoftmaxKernel(const T* input,
                              T* output,
                              int64_t rows,
                              int64_t dim,
                              int64_t inner) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t row_index = thread_id; row_index < rows; row_index += total_threads) {
    const int64_t outer_index = row_index / inner;
    const int64_t inner_index = row_index - outer_index * inner;
    const int64_t base = outer_index * dim * inner + inner_index;

    AccT max_value = -INFINITY;
    for (int64_t d = 0; d < dim; ++d) {
      const AccT value = SoftmaxToAccum<AccT>(input[base + d * inner]);
      max_value = value > max_value ? value : max_value;
    }

    AccT sum = 0;
    for (int64_t d = 0; d < dim; ++d) {
      const AccT value =
          SoftmaxExp(SoftmaxToAccum<AccT>(input[base + d * inner]) -
                     max_value);
      sum += value;
    }

    const AccT inv_sum = static_cast<AccT>(1) / sum;
    for (int64_t d = 0; d < dim; ++d) {
      const AccT value =
          SoftmaxExp(SoftmaxToAccum<AccT>(input[base + d * inner]) -
                     max_value) *
          inv_sum;
      output[base + d * inner] = SoftmaxFromAccum<T, AccT>(value);
    }
  }
}

}  // namespace

template <typename T, typename AccT>
musaError_t LaunchMusaSoftmaxTyped(const void* input,
                                   void* output,
                                   int64_t outer,
                                   int64_t dim,
                                   int64_t inner,
                                   musaStream_t stream) {
  const int64_t rows = outer * inner;
  if (rows == 0 || dim == 0) {
    return musaSuccess;
  }
  SoftmaxKernel<T, AccT><<<BlocksForCount(rows), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), rows,
      dim, inner);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}

musaError_t LaunchMusaSoftmaxKernel(const void* input,
                                    void* output,
                                    int64_t outer,
                                    int64_t dim,
                                    int64_t inner,
                                    MusaElementType elem_type,
                                    musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchMusaSoftmaxTyped<float, float>(input, output, outer, dim,
                                                  inner, stream);
    case MusaElementType::Double:
      return LaunchMusaSoftmaxTyped<double, double>(input, output, outer, dim,
                                                    inner, stream);
    case MusaElementType::Float16:
      return LaunchMusaSoftmaxTyped<__half, float>(input, output, outer, dim,
                                                   inner, stream);
    case MusaElementType::BFloat16:
      return LaunchMusaSoftmaxTyped<__mt_bfloat16, float>(
          input, output, outer, dim, inner, stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaSoftmaxFloatKernel(const float* input,
                                         float* output,
                                         int64_t outer,
                                         int64_t dim,
                                         int64_t inner,
                                         musaStream_t stream) {
  return LaunchMusaSoftmaxKernel(input, output, outer, dim, inner,
                                 MusaElementType::Float, stream);
}
