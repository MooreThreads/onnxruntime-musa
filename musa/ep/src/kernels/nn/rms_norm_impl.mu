#include "nn/rms_norm_impl.h"

#include "shared_inc/musa_kernel_common.mu.h"

#include <stdint.h>

namespace {

template <typename T>
__global__ void RmsNormKernel(const T* input,
                              const float* gamma,
                              T* output,
                              int64_t rows,
                              int64_t norm_size,
                              float epsilon) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  float sum_square = 0.0f;
  const int64_t row_offset = row * norm_size;
  for (int64_t col = threadIdx.x; col < norm_size; col += blockDim.x) {
    const float value = MusaScalarToFloat(input[row_offset + col]);
    sum_square += value * value;
  }

  __shared__ float shared[kThreadsPerBlock];
  shared[threadIdx.x] = sum_square;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_rms =
      rsqrtf(shared[0] / static_cast<float>(norm_size) + epsilon);
  for (int64_t col = threadIdx.x; col < norm_size; col += blockDim.x) {
    const float value =
        MusaScalarToFloat(input[row_offset + col]) * inv_rms * gamma[col];
    output[row_offset + col] = MusaScalarFromFloat<T>(value);
  }
}

template <typename T>
musaError_t LaunchRmsNormTyped(const void* input,
                               const float* gamma,
                               void* output,
                               int64_t rows,
                               int64_t norm_size,
                               float epsilon,
                               musaStream_t stream) {
  if (rows == 0 || norm_size == 0) {
    return musaSuccess;
  }
  if (rows > INT32_MAX || norm_size > INT32_MAX) {
    return musaErrorNotSupported;
  }
  RmsNormKernel<T><<<static_cast<int>(rows), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input), gamma, reinterpret_cast<T*>(output),
      rows, norm_size, epsilon);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaRmsNormKernel(const void* input,
                                    const float* gamma,
                                    void* output,
                                    int64_t rows,
                                    int64_t norm_size,
                                    float epsilon,
                                    MusaElementType elem_type,
                                    musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchRmsNormTyped<float>(input, gamma, output, rows, norm_size,
                                       epsilon, stream);
    case MusaElementType::Float16:
      return LaunchRmsNormTyped<__half>(input, gamma, output, rows, norm_size,
                                        epsilon, stream);
    case MusaElementType::BFloat16:
      return LaunchRmsNormTyped<__mt_bfloat16>(input, gamma, output, rows,
                                               norm_size, epsilon, stream);
    case MusaElementType::Double:
      return LaunchRmsNormTyped<double>(input, gamma, output, rows, norm_size,
                                        epsilon, stream);
    default:
      return musaErrorNotSupported;
  }
}
