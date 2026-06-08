#pragma once

#include <cstdint>

#include <musa_runtime.h>

musaError_t LaunchMusaRawCopy3Float(
    const float* input0, const float* input1, const float* input2,
    float* output0, float* output1, float* output2, int64_t element_count,
    musaStream_t stream);
