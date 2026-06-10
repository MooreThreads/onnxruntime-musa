#pragma once

#include <musa_runtime.h>

enum class MusaSplitReduceMode : int32_t {
  Prod = 0,
  Mean = 1,
};

musaError_t LaunchMusaSplitReduce2Float(
    const float* input, float* output0, float* output1, int64_t batch,
    int64_t axis_dim, int64_t inner, int64_t offset0, int64_t width0,
    MusaSplitReduceMode mode0, int64_t offset1, int64_t width1,
    MusaSplitReduceMode mode1, musaStream_t stream);
