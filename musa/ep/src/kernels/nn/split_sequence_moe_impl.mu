// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

#include "split_sequence_moe_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void SplitSequenceMoESkipLayerNormFloatKernel(
    const float* residual_input, const float* ffn_output, const float* gamma,
    const float* beta, const int64_t* group_offsets, float* output,
    int64_t rows, int64_t group_count, int64_t width, float epsilon) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  int64_t group = 0;
  while (group + 1 < group_count && row >= group_offsets[group + 1]) {
    ++group;
  }
  const int64_t group_base = group * width;
  const int64_t row_base = row * width;

  float sum = 0.0f;
  for (int64_t column = threadIdx.x; column < width;
       column += blockDim.x) {
    sum += residual_input[row_base + column] + ffn_output[row_base + column];
  }

  __shared__ float shared[kThreadsPerBlock];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = shared[0] / static_cast<float>(width);

  float variance_sum = 0.0f;
  for (int64_t column = threadIdx.x; column < width;
       column += blockDim.x) {
    const float value = residual_input[row_base + column] +
                        ffn_output[row_base + column] - mean;
    variance_sum += value * value;
  }
  shared[threadIdx.x] = variance_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_std =
      rsqrtf(shared[0] / static_cast<float>(width) + epsilon);

  for (int64_t column = threadIdx.x; column < width;
       column += blockDim.x) {
    const float value = residual_input[row_base + column] +
                        ffn_output[row_base + column];
    const float normalized = (value - mean) * inv_std;
    output[row_base + column] =
        normalized * gamma[group_base + column] + beta[group_base + column];
  }
}

}  // namespace

musaError_t LaunchSplitSequenceMoESkipLayerNormFloatKernel(
    const float* residual_input, const float* ffn_output,
    const float* gamma, const float* beta, const int64_t* group_offsets,
    float* output, int64_t rows, int64_t group_count, int64_t width,
    float epsilon, musaStream_t stream) {
  if (rows == 0 || width == 0) {
    return musaSuccess;
  }
  if (rows < 0 || group_count < 1 || width < 1 || rows > INT32_MAX) {
    return musaErrorNotSupported;
  }
  SplitSequenceMoESkipLayerNormFloatKernel<<<static_cast<int>(rows),
                                             kThreadsPerBlock, 0, stream>>>(
      residual_input, ffn_output, gamma, beta, group_offsets, output, rows,
      group_count, width, epsilon);
  return musaGetLastError();
}
