#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaReplaceInvalidIdInt32Kernel(
    const int32_t* input, int32_t* output, int64_t count, int32_t threshold,
    int32_t replacement, musaStream_t stream);

musaError_t LaunchMusaReplaceInvalidIdInt64Kernel(
    const int64_t* input, int64_t* output, int64_t count, int64_t threshold,
    int64_t replacement, musaStream_t stream);
