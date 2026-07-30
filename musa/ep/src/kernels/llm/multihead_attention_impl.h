#pragma once

#include <musa_runtime.h>

#include <cstdint>

musaError_t LaunchMusaPackQkvBiasKernel(const float* query, const float* key,
                                        const float* value, const float* bias,
                                        float* packed_qkv, int64_t token_count,
                                        int64_t hidden_size,
                                        musaStream_t stream);
