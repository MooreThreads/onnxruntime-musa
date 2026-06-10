#pragma once

#include "shared_inc/device_kernel_types.h"

struct MusaConcatSplitSegment {
  const void* input;
  void* output;
  int64_t input_cols;
  int64_t output_cols;
  int64_t source_offset;
  int64_t width;
  int64_t dst_offset;
};

struct MusaConcatSplitSumOutput {
  float* output;
  int64_t output_cols;
  int64_t term_offset;
  int64_t term_count;
};

struct MusaConcatSplitSumTerm {
  const float* input;
  int64_t input_cols;
  int64_t source_offset;
};

struct MusaConcatSplitCopyBlock {
  int64_t segment_index;
  int64_t element_offset;
};

musaError_t LaunchMusaConcatSplitBatchedCopy(
    const MusaConcatSplitSegment* device_segments, int64_t segment_count,
    const MusaConcatSplitCopyBlock* device_blocks, int64_t block_count,
    int64_t rows, int32_t element_size, musaStream_t stream);

musaError_t LaunchMusaConcatSplitSums(
    const MusaConcatSplitSumOutput* device_outputs,
    const MusaConcatSplitSumTerm* device_terms, int64_t output_count,
    int64_t rows, int64_t max_width, musaStream_t stream);
