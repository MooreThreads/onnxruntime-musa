#pragma once

#include <musa_runtime.h>

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaCenteredReduceFloatKernel(
    const float* input, float* first_reduce, float* second_reduce, int64_t rows,
    int64_t inner, MusaReduceOp first_op, MusaReduceOp second_op,
    musaStream_t stream);
