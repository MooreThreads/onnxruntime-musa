#pragma once

#include <musa_runtime.h>

#include <cstdint>

musaError_t LaunchMusaCumSumKernel(const void* input, void* output,
                                   int32_t elem_type, int64_t output_size,
                                   int64_t axis_dim, int64_t axis_stride,
                                   bool exclusive, bool reverse,
                                   musaStream_t stream);
