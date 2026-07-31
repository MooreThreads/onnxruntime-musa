// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

#include "moe_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void MoEBiasReluFloatKernel(float* values, const float* biases,
                                       int64_t rows, int64_t expert_count,
                                       int64_t width) {
  const int64_t total = rows * expert_count * width;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += stride) {
    const int64_t expert = index / (rows * width);
    const int64_t column = index % width;
    const float value = values[index] + biases[expert * width + column];
    values[index] = value > 0.0f ? value : 0.0f;
  }
}

__global__ void MoERouterReduceMultiFloatKernel(
    const float* expert_values, const float* biases,
    const float* const* routers, float* const* outputs, int64_t rows,
    int64_t expert_count, int64_t output_width, int64_t router_count) {
  const int64_t output_elements = rows * output_width;
  const int64_t total = router_count * output_elements;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += stride) {
    const int64_t router_index = index / output_elements;
    const int64_t output_index = index - router_index * output_elements;
    const int64_t row = output_index / output_width;
    const int64_t column = output_index - row * output_width;
    const float* router = routers[router_index];
    float sum = 0.0f;
    for (int64_t expert = 0; expert < expert_count; ++expert) {
      const int64_t expert_index =
          (expert * rows + row) * output_width + column;
      const float biased =
          expert_values[expert_index] + biases[expert * output_width + column];
      const float activated = biased > 0.0f ? biased : 0.0f;
      sum += activated * router[row * expert_count + expert];
    }
    outputs[router_index][output_index] = sum;
  }
}

}  // namespace

musaError_t LaunchMoEBiasReluFloatKernel(float* values, const float* biases,
                                         int64_t rows, int64_t expert_count,
                                         int64_t width, musaStream_t stream) {
  const int64_t total = rows * expert_count * width;
  if (total == 0) {
    return musaSuccess;
  }
  MoEBiasReluFloatKernel<<<BlocksForCount(total), kThreadsPerBlock, 0,
                           stream>>>(values, biases, rows, expert_count, width);
  return musaGetLastError();
}

musaError_t LaunchMoERouterReduceMultiFloatKernel(
    const float* expert_values, const float* biases,
    const float* const* routers, float* const* outputs, int64_t rows,
    int64_t expert_count, int64_t output_width, int64_t router_count,
    musaStream_t stream) {
  const int64_t total = router_count * rows * output_width;
  if (total == 0) {
    return musaSuccess;
  }
  MoERouterReduceMultiFloatKernel<<<BlocksForCount(total), kThreadsPerBlock, 0,
                                    stream>>>(expert_values, biases, routers,
                                              outputs, rows, expert_count,
                                              output_width, router_count);
  return musaGetLastError();
}
