#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaMaskedGatherReduceFloatKernel(const uint8_t* mask,
                                                    const float* data,
                                                    float* output,
                                                    int64_t count,
                                                    MusaReduceOp op,
                                                    musaStream_t stream);
