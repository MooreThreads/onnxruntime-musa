#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaRecRankCalibrationKernel(
    const double* task_relu, const double* task_scores, float* output,
    int64_t rows, int64_t bucket_size, float epsilon,
    bool complement_from_clipped_probability, musaStream_t stream);
