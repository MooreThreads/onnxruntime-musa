#pragma once

#include <musa_runtime.h>

#include <cstdint>

musaError_t LaunchMusaParallelEinsumActivationPackW1Kernel(
    const float* w0, const float* w1, const float* w2, const float* w3,
    float* packed, int64_t input_dim, int64_t hidden_dim, musaStream_t stream);

musaError_t LaunchMusaParallelEinsumActivationStage23Kernel(
    const float* stage1, const float* gate_input, const float* w2_0,
    const float* w2_1, const float* w2_2, const float* w2_3, const float* w3_0,
    const float* w3_1, const float* w3_2, const float* w3_3, const float* bias,
    float* output, int64_t batch, int64_t input_dim, int64_t hidden_dim,
    musaStream_t stream);
