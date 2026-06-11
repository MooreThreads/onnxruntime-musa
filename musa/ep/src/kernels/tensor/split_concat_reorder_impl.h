#pragma once

#include <musa_runtime.h>

#include <cstdint>

musaError_t LaunchMusaSplitConcatReorderFloat(const float* input, float* output,
                                              int64_t batch, int64_t sequence,
                                              int64_t part_count,
                                              int64_t part_width,
                                              musaStream_t stream);
