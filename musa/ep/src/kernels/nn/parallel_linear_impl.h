#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchParallelLinearPostFloatKernel(
    const float* merged, float* const* outputs, const float* const* biases,
    const int64_t* branch_widths, const int64_t* branch_offsets, int64_t rows,
    int64_t branch_count, int64_t total_width, MusaUnaryOp activation,
    bool has_activation, float activation_alpha, musaStream_t stream);
