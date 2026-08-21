#pragma once

#include <cstdint>

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchGenerateMTGRCustomMaskKernel(
    bool* mask, int64_t* total_length_output, int64_t* target_length_output,
    int64_t upstream_length, int64_t target_length, int64_t capacity,
    musaStream_t stream);

musaError_t LaunchGenerateMTGRCroppedMaskKernel(
    int32_t* mask, int64_t* target_length_output, int64_t upstream_length,
    int64_t target_length, int64_t dimension, musaStream_t stream);
