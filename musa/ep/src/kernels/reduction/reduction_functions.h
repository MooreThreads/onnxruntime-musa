#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaReduceFloatKernel(const float* input, float* output,
                                        MusaReduceParams params,
                                        MusaReduceOp op, musaStream_t stream);

musaError_t LaunchMusaReduceInt32Kernel(const int32_t* input, int32_t* output,
                                        MusaReduceParams params,
                                        MusaReduceOp op, musaStream_t stream);

musaError_t LaunchMusaReduceInt64Kernel(const int64_t* input, int64_t* output,
                                        MusaReduceParams params,
                                        MusaReduceOp op, musaStream_t stream);
