#include <stdint.h>

#include "nn/rms_norm_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

constexpr int kWarpSize = 32;

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

__device__ __forceinline__ float CastRmsNormSquare(float value) {
  return value * value;
}

template <int kBlockThreads>
__device__ __forceinline__ float CastRmsNormBlockReduceSum(float value) {
  __shared__ float shared[kBlockThreads / kWarpSize];
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x >> 5;

#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }

  if (lane == 0) {
    shared[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    value = threadIdx.x < (kBlockThreads / kWarpSize) ? shared[threadIdx.x]
                                                      : 0.0f;
#pragma unroll
    for (int mask = (kBlockThreads / kWarpSize) / 2; mask > 0; mask >>= 1) {
      value += __shfl_xor_sync(0xffffffff, value, mask);
    }
    if (lane == 0) {
      shared[0] = value;
    }
  }
  __syncthreads();
  return shared[0];
}

template <int kGroupSize>
__device__ __forceinline__ float CastRmsNormGroupReduceSum(float value,
                                                           int group,
                                                           int lane) {
  constexpr int kWarpsPerGroup = kGroupSize / kWarpSize;
  __shared__ float shared[kThreadsPerBlock / kWarpSize];
  const int warp_lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x >> 5;

#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }

  if (warp_lane == 0) {
    shared[warp_id] = value;
  }
  __syncthreads();

  value = lane < kWarpsPerGroup ? shared[group * kWarpsPerGroup + lane] : 0.0f;
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }
  if (lane == 0) {
    shared[group * kWarpsPerGroup] = value;
  }
  __syncthreads();
  return shared[group * kWarpsPerGroup];
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

template <int kBlockThreads>
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
    sum_square += CastRmsNormSquare(value);
  }

  const float mean =
      CastRmsNormBlockReduceSum<kBlockThreads>(sum_square) /
      static_cast<float>(norm_size);
  WriteCastRmsNormBf16Row(input, gamma, output, row_offset, norm_size, mean,
                          epsilon, threadIdx.x, blockDim.x);
}

template <int kGroupSize>
__global__ void CastRmsNormBf16LastAxisMultiOutputBlockKernel(
    const __mt_bfloat16* input, const __mt_bfloat16* gamma,
    __mt_bfloat16* output, int64_t rows, int64_t norm_size, float epsilon) {
  constexpr int kOutputsPerBlock = kThreadsPerBlock / kGroupSize;
  const int group = threadIdx.x / kGroupSize;
  const int lane = threadIdx.x - group * kGroupSize;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kOutputsPerBlock + group;
  const bool valid = row < rows;

  float sum_square = 0.0f;
  const int64_t row_offset = row * norm_size;
  if (valid) {
    for (int64_t col = lane; col < norm_size; col += kGroupSize) {
      const float value = MusaScalarToFloat(input[row_offset + col]);
      sum_square += CastRmsNormSquare(value);
    }
  }

  const float group_sum =
      CastRmsNormGroupReduceSum<kGroupSize>(sum_square, group, lane);
  if (valid) {
    const float mean = group_sum / static_cast<float>(norm_size);
    WriteCastRmsNormBf16Row(input, gamma, output, row_offset, norm_size, mean,
                            epsilon, lane, kGroupSize);
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
      sum_square += CastRmsNormSquare(value);
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
    if (rows >= 1024 && norm_size <= 1024) {
      if (norm_size <= 128) {
        constexpr int kGroupSize = 32;
        constexpr int kOutputsPerBlock = kThreadsPerBlock / kGroupSize;
        CastRmsNormBf16LastAxisMultiOutputBlockKernel<kGroupSize><<<
            static_cast<int>((rows + kOutputsPerBlock - 1) / kOutputsPerBlock),
            kThreadsPerBlock, 0, stream>>>(
            typed_input, typed_gamma, typed_output, rows, norm_size, epsilon);
      } else if (norm_size <= 256) {
        constexpr int kGroupSize = 64;
        constexpr int kOutputsPerBlock = kThreadsPerBlock / kGroupSize;
        CastRmsNormBf16LastAxisMultiOutputBlockKernel<kGroupSize><<<
            static_cast<int>((rows + kOutputsPerBlock - 1) / kOutputsPerBlock),
            kThreadsPerBlock, 0, stream>>>(
            typed_input, typed_gamma, typed_output, rows, norm_size, epsilon);
      } else {
        constexpr int kGroupSize = 128;
        constexpr int kOutputsPerBlock = kThreadsPerBlock / kGroupSize;
        CastRmsNormBf16LastAxisMultiOutputBlockKernel<kGroupSize><<<
            static_cast<int>((rows + kOutputsPerBlock - 1) / kOutputsPerBlock),
            kThreadsPerBlock, 0, stream>>>(
            typed_input, typed_gamma, typed_output, rows, norm_size, epsilon);
      }
      return musaGetLastError();
    }
    if (norm_size <= 512) {
      constexpr int kBlockThreads = 128;
      CastRmsNormBf16LastAxisBlockKernel<kBlockThreads>
          <<<static_cast<int>(rows), kBlockThreads, 0, stream>>>(
              typed_input, typed_gamma, typed_output, rows, norm_size, epsilon);
    } else {
      constexpr int kBlockThreads = kThreadsPerBlock;
      CastRmsNormBf16LastAxisBlockKernel<kBlockThreads>
          <<<static_cast<int>(rows), kBlockThreads, 0, stream>>>(
              typed_input, typed_gamma, typed_output, rows, norm_size, epsilon);
    }
    return musaGetLastError();
  }
  CastRmsNormBf16SingleAxisKernel<<<BlocksForCount(rows), kThreadsPerBlock, 0,
                                    stream>>>(
      typed_input, typed_gamma, typed_output, rows, norm_size, epsilon);
  return musaGetLastError();
}
