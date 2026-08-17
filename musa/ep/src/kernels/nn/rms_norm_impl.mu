#include <stdint.h>

#include "nn/rms_norm_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T>
__global__ void RmsNormKernel(const T* input, const T* gamma, T* output,
                              int64_t rows, int64_t norm_size, float epsilon) {
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
    const float value = MusaScalarToFloat(input[row_offset + col]) * inv_rms *
                        MusaScalarToFloat(gamma[col]);
    output[row_offset + col] = MusaScalarFromFloat<T>(value);
  }
}

template <typename T>
musaError_t LaunchRmsNormTyped(const void* input, const void* gamma,
                               void* output, int64_t rows, int64_t norm_size,
                               float epsilon, musaStream_t stream) {
  if (rows == 0 || norm_size == 0) {
    return musaSuccess;
  }
  if (rows > INT32_MAX || norm_size > INT32_MAX) {
    return musaErrorNotSupported;
  }
  RmsNormKernel<T><<<static_cast<int>(rows), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input), reinterpret_cast<const T*>(gamma),
      reinterpret_cast<T*>(output), rows, norm_size, epsilon);
  return musaGetLastError();
}

__device__ __forceinline__ float CastRmsNormPow2(float value) {
  return powf(value, 2.0f);
}

__device__ __forceinline__ void WriteCastRmsNormBf16Row(
    const __mt_bfloat16* input, const __mt_bfloat16* gamma,
    __mt_bfloat16* output, int64_t row_offset, int64_t norm_size, float mean,
    float epsilon, int64_t start_col, int64_t col_stride) {
  const float denom = sqrtf(mean + epsilon);
  for (int64_t col = start_col; col < norm_size; col += col_stride) {
    const float normalized = MusaScalarToFloat(input[row_offset + col]) / denom;
    const __mt_bfloat16 rounded_normalized =
        MusaScalarFromFloat<__mt_bfloat16>(normalized);
    const float value =
        MusaScalarToFloat(rounded_normalized) * MusaScalarToFloat(gamma[col]);
    output[row_offset + col] = MusaScalarFromFloat<__mt_bfloat16>(value);
  }
}

__global__ void CastRmsNormBf16LastAxisBlockKernel(
    const __mt_bfloat16* input, const __mt_bfloat16* gamma,
    __mt_bfloat16* output, int64_t rows, int64_t norm_size, float epsilon) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  float sum_square = 0.0f;
  const int64_t row_offset = row * norm_size;
  for (int64_t col = threadIdx.x; col < norm_size; col += blockDim.x) {
    const float value = MusaScalarToFloat(input[row_offset + col]);
    sum_square += CastRmsNormPow2(value);
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

  const float mean = shared[0] / static_cast<float>(norm_size);
  WriteCastRmsNormBf16Row(input, gamma, output, row_offset, norm_size, mean,
                          epsilon, threadIdx.x, blockDim.x);
}

__global__ void CastRmsNormBf16LastAxisMultiOutputBlockKernel(
    const __mt_bfloat16* input, const __mt_bfloat16* gamma,
    __mt_bfloat16* output, int64_t rows, int64_t norm_size, float epsilon,
    int group_size, int outputs_per_block) {
  const int group = threadIdx.x / group_size;
  const int lane = threadIdx.x - group * group_size;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * outputs_per_block + group;
  const bool valid = row < rows;

  float sum_square = 0.0f;
  const int64_t row_offset = row * norm_size;
  if (valid) {
    for (int64_t col = lane; col < norm_size; col += group_size) {
      const float value = MusaScalarToFloat(input[row_offset + col]);
      sum_square += CastRmsNormPow2(value);
    }
  }

  __shared__ float shared[kThreadsPerBlock];
  shared[threadIdx.x] = sum_square;
  __syncthreads();
  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (valid) {
    const float mean =
        shared[threadIdx.x - lane] / static_cast<float>(norm_size);
    WriteCastRmsNormBf16Row(input, gamma, output, row_offset, norm_size, mean,
                            epsilon, lane, group_size);
  }
}

__global__ void CastRmsNormBf16SingleAxisKernel(const __mt_bfloat16* input,
                                                const __mt_bfloat16* gamma,
                                                __mt_bfloat16* output,
                                                int64_t rows, int64_t norm_size,
                                                float epsilon) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t row = thread_id; row < rows; row += total_threads) {
    const int64_t row_offset = row * norm_size;
    float sum_square = 0.0f;
    for (int64_t col = 0; col < norm_size; ++col) {
      const float value = MusaScalarToFloat(input[row_offset + col]);
      sum_square += CastRmsNormPow2(value);
    }
    const float mean = sum_square / static_cast<float>(norm_size);
    WriteCastRmsNormBf16Row(input, gamma, output, row_offset, norm_size, mean,
                            epsilon, 0, 1);
  }
}

}  // namespace

musaError_t LaunchMusaRmsNormKernel(const void* input, const void* gamma,
                                    void* output, int64_t rows,
                                    int64_t norm_size, float epsilon,
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

musaError_t LaunchMusaCastRmsNormBf16Kernel(const void* input,
                                            const void* gamma, void* output,
                                            int64_t rows, int64_t norm_size,
                                            float epsilon,
                                            musaStream_t stream) {
  if (rows == 0 || norm_size == 0) {
    return musaSuccess;
  }
  if (rows > INT32_MAX || norm_size > INT32_MAX) {
    return musaErrorNotSupported;
  }
  const auto* typed_input = reinterpret_cast<const __mt_bfloat16*>(input);
  const auto* typed_gamma = reinterpret_cast<const __mt_bfloat16*>(gamma);
  auto* typed_output = reinterpret_cast<__mt_bfloat16*>(output);
  if (norm_size >= 64) {
    if (norm_size <= 256 && rows >= 1024) {
      const int group_size = 64;
      const int outputs_per_block = kThreadsPerBlock / group_size;
      CastRmsNormBf16LastAxisMultiOutputBlockKernel<<<
          static_cast<int>((rows + outputs_per_block - 1) / outputs_per_block),
          kThreadsPerBlock, 0, stream>>>(typed_input, typed_gamma, typed_output,
                                         rows, norm_size, epsilon, group_size,
                                         outputs_per_block);
      return musaGetLastError();
    }
    CastRmsNormBf16LastAxisBlockKernel<<<static_cast<int>(rows),
                                         kThreadsPerBlock, 0, stream>>>(
        typed_input, typed_gamma, typed_output, rows, norm_size, epsilon);
    return musaGetLastError();
  }
  CastRmsNormBf16SingleAxisKernel<<<BlocksForCount(rows), kThreadsPerBlock, 0,
                                    stream>>>(
      typed_input, typed_gamma, typed_output, rows, norm_size, epsilon);
  return musaGetLastError();
}
