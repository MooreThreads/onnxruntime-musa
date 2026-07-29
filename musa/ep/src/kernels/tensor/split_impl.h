#pragma once

#include "shared_inc/device_kernel_types.h"

struct MusaSplitCopyDesc {
  void* output;
  int64_t axis_start;
  int64_t split_size;
};

struct MusaSplitElementDesc {
  void* output;
  int64_t output_width;
  int64_t local_element;
};

musaError_t LaunchMusaSplitCopies(const void* input, void* const* outputs,
                                  const int64_t* split_sizes,
                                  int64_t output_count, int64_t outer,
                                  int64_t inner, int64_t input_axis,
                                  int32_t element_size, musaStream_t stream);

musaError_t LaunchMusaSplitManySmallCopies(const void* input,
                                           const MusaSplitCopyDesc* descriptors,
                                           int64_t output_count, int64_t outer,
                                           int64_t inner, int64_t input_axis,
                                           int32_t element_size,
                                           musaStream_t stream);

musaError_t LaunchMusaSplitManySmallRows(
    const void* input, const MusaSplitElementDesc* descriptors, int64_t outer,
    int64_t input_row_elements, int32_t element_size, musaStream_t stream);
