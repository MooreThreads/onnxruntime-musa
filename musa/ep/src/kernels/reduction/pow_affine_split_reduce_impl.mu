#include "reduction/pow_affine_split_reduce_impl.h"

#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ float ApplyAffine(float value, float affine,
                                             MusaPowAffineOp op) {
  return op == MusaPowAffineOp::Sub ? value - affine : value + affine;
}

__device__ __forceinline__ int64_t BroadcastIndex(
    int64_t row, int64_t axis, int64_t col, const int64_t strides[3]) {
  return row * strides[0] + axis * strides[1] + col * strides[2];
}

__device__ __forceinline__ float ReducePowAffineSegment(
    const float* input, const float* exponent, const float* affine,
    float* affine_output, int64_t row, int64_t col, int64_t offset,
    int64_t width, const MusaPowAffineSplitReduceParams& params,
    MusaSplitReduceMode mode) {
  float acc = mode == MusaSplitReduceMode::Prod ? 1.0f : 0.0f;
  for (int64_t i = 0; i < width; ++i) {
    const int64_t axis = offset + i;
    const int64_t input_index =
        row * params.axis_dim * params.inner + axis * params.inner + col;
    const int64_t exponent_index =
        BroadcastIndex(row, axis, col, params.exponent_strides);
    const int64_t affine_index =
        BroadcastIndex(row, axis, col, params.affine_strides);
    const float value =
        ApplyAffine(powf(input[input_index], exponent[exponent_index]),
                    affine[affine_index], params.affine_op);
    if (affine_output != nullptr) {
      affine_output[input_index] = value;
    }
    if (mode == MusaSplitReduceMode::Prod) {
      acc *= value;
    } else {
      acc += value;
    }
  }
  if (mode == MusaSplitReduceMode::Mean) {
    acc /= static_cast<float>(width);
  }
  return acc;
}

__global__ void PowAffineSplitReduce2FloatKernel(
    const float* input, const float* exponent, const float* affine,
    float* affine_output, float* output0, float* output1,
    MusaPowAffineSplitReduceParams params) {
  const int64_t total = params.batch * params.inner;
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < total; index += total_threads) {
    const int64_t row = index / params.inner;
    const int64_t col = index - row * params.inner;
    output0[index] = ReducePowAffineSegment(
        input, exponent, affine, affine_output, row, col, params.offset0, params.width0,
        params, params.mode0);
    output1[index] = ReducePowAffineSegment(
        input, exponent, affine, affine_output, row, col, params.offset1, params.width1,
        params, params.mode1);
  }
}

}  // namespace

musaError_t LaunchMusaPowAffineSplitReduce2Float(
    const float* input, const float* exponent, const float* affine,
    float* affine_output, float* output0, float* output1,
    MusaPowAffineSplitReduceParams params, musaStream_t stream) {
  if (params.batch == 0 || params.inner == 0) {
    return musaSuccess;
  }
  PowAffineSplitReduce2FloatKernel<<<BlocksForCount(params.batch * params.inner),
                                     kThreadsPerBlock, 0, stream>>>(
      input, exponent, affine, affine_output, output0, output1, params);
  return musaGetLastError();
}
