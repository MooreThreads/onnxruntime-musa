#pragma once

#include "shared_inc/device_kernel_types.h"

#include <math.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <stdint.h>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 4096;

int BlocksForCount(int64_t count) {
  int64_t blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (blocks > kMaxBlocks) {
    blocks = kMaxBlocks;
  }
  return static_cast<int>(blocks);
}

__device__ __forceinline__ void ResolveBroadcastIndices(
    int64_t index,
    const MusaBroadcastParams& params,
    int64_t& lhs_index,
    int64_t& rhs_index) {
  lhs_index = 0;
  rhs_index = 0;
  int64_t remaining = index;
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    const int64_t coord = remaining / params.output_strides[dim];
    remaining -= coord * params.output_strides[dim];
    lhs_index += coord * params.lhs_strides[dim];
    rhs_index += coord * params.rhs_strides[dim];
  }
}

template <typename T>
__device__ __forceinline__ float MusaScalarToFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float MusaScalarToFloat<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float MusaScalarToFloat<__mt_bfloat16>(
    __mt_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ double MusaScalarToDouble(T value) {
  return static_cast<double>(value);
}

template <>
__device__ __forceinline__ double MusaScalarToDouble<__half>(__half value) {
  return static_cast<double>(__half2float(value));
}

template <>
__device__ __forceinline__ double MusaScalarToDouble<__mt_bfloat16>(
    __mt_bfloat16 value) {
  return static_cast<double>(__bfloat162float(value));
}

template <typename T>
__device__ __forceinline__ T MusaScalarFromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half MusaScalarFromFloat<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __mt_bfloat16
MusaScalarFromFloat<__mt_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
__device__ __forceinline__ T MusaScalarFromDouble(double value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half MusaScalarFromDouble<__half>(double value) {
  return __float2half_rn(static_cast<float>(value));
}

template <>
__device__ __forceinline__ __mt_bfloat16
MusaScalarFromDouble<__mt_bfloat16>(double value) {
  return __float2bfloat16_rn(static_cast<float>(value));
}

}  // namespace
