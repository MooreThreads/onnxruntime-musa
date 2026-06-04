#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaReduceFloatKernel(const float* input,
                                        float* output,
                                        MusaReduceParams params,
                                        MusaReduceOp op,
                                        musaStream_t stream);
