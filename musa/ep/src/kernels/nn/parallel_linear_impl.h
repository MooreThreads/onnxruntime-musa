#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchParallelLinearPostFloatKernel(
    const float* merged, float* const* outputs, const float* const* biases,
    int64_t rows, int64_t branch_count, int64_t branch_width,
    MusaUnaryOp activation, bool has_activation, float activation_alpha,
    musaStream_t stream);

musaError_t LaunchParallelLinearPostDirectFloatKernel(
    const float* merged, float* output0, float* output1, float* output2,
    const float* bias0, const float* bias1, const float* bias2, int64_t rows,
    int64_t branch_count, int64_t branch_width, MusaUnaryOp activation,
    bool has_activation, float activation_alpha, musaStream_t stream);

musaError_t LaunchParallelLinearGatedMlpPostFloatKernel(
    const float* merged, float* output, const float* gate_bias,
    const float* up_bias, int64_t rows, int64_t branch_width,
    musaStream_t stream);
