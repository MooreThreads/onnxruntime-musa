#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchParallelLinearPostFloatKernel(
    const float* merged, float* const* outputs, const float* const* biases,
    int64_t rows, int64_t branch_count, int64_t branch_width,
    MusaUnaryOp activation, bool has_activation, float activation_alpha,
    musaStream_t stream);
