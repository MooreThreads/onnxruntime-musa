#pragma once

#include <musa_runtime.h>

#include "reduction/split_reduce_impl.h"

enum class MusaPowAffineOp : int32_t {
  Add = 0,
  Sub = 1,
};

struct MusaPowAffineSplitReduceParams {
  int64_t batch;
  int64_t axis_dim;
  int64_t inner;
  int64_t exponent_strides[3];
  int64_t affine_strides[3];
  int64_t offset0;
  int64_t width0;
  MusaSplitReduceMode mode0;
  int64_t offset1;
  int64_t width1;
  MusaSplitReduceMode mode1;
  MusaPowAffineOp affine_op;
};

musaError_t LaunchMusaPowAffineSplitReduce2Float(
    const float* input, const float* exponent, const float* affine,
    float* affine_output, float* output0, float* output1,
    MusaPowAffineSplitReduceParams params, musaStream_t stream);
