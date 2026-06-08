#pragma once

#include <cstdint>

#include <musa_runtime.h>

constexpr int32_t kMusaSliceSumConcatMaxSegments = 16;
constexpr int32_t kMusaSliceSumConcatMaxSlices = 128;

enum class MusaSliceSumConcatSegmentMode : int32_t {
  Direct = 0,
  SumSlices = 1,
};

struct MusaSliceSumConcatSlice {
  const float* input;
  int64_t input_cols;
  int64_t start_col;
};

struct MusaSliceSumConcatSegment {
  int32_t mode;
  int32_t slice_start;
  int32_t slice_count;
  int64_t dst_col;
  int64_t width;
  const float* direct_input;
  int64_t direct_input_cols;
};

struct MusaSliceSumConcatParams {
  int64_t batch;
  int64_t output_cols;
  int32_t segment_count;
  int32_t slice_count;
  MusaSliceSumConcatSegment segments[kMusaSliceSumConcatMaxSegments];
  MusaSliceSumConcatSlice slices[kMusaSliceSumConcatMaxSlices];
};

musaError_t LaunchMusaSliceSumConcatFloat(const MusaSliceSumConcatParams& params,
                                          float* output,
                                          musaStream_t stream);
