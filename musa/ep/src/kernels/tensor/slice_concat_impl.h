#pragma once

#include <cstdint>

#include <musa_runtime.h>

struct MusaSliceConcatSegment {
  const float* input;
  int64_t input_cols;
  int64_t start_col;
  int64_t width;
  int64_t dst_offset;
  int64_t block_offset;
  int64_t block_count;
  int32_t zero_fill;
};

constexpr int kSliceConcatThreadsPerBlock = 256;

musaError_t LaunchMusaSliceConcatKernel(
    const MusaSliceConcatSegment* device_segments, int64_t segment_count,
    float* output, int64_t rows, int64_t output_cols, musaStream_t stream);

musaError_t LaunchMusaSliceConcatSegmentedKernel(
    const MusaSliceConcatSegment* device_segments, int64_t segment_count,
    int64_t total_blocks, float* output, int64_t rows, int64_t output_cols,
    musaStream_t stream);

musaError_t LaunchMusaSliceConcatEqualWidthKernel(
    const MusaSliceConcatSegment* device_segments, int64_t segment_count,
    float* output, int64_t rows, int64_t output_cols, int64_t segment_width,
    int32_t width_shift, musaStream_t stream);
