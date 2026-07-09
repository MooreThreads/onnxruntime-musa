#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaSparseIdToMaskInt32Kernel(
    const int32_t* dense_ids, const int32_t* bound_ids,
    const int32_t* sparse_ids, float* output, int64_t output_count,
    int64_t candidate_count, int64_t sparse_count, int32_t default_id,
    musaStream_t stream);

musaError_t LaunchMusaSparseIdToMaskInt32Kernel(
    const int32_t* dense_ids, const int32_t* bound_ids,
    const int32_t* sparse_ids, int32_t* output, int64_t output_count,
    int64_t candidate_count, int64_t sparse_count, int32_t default_id,
    musaStream_t stream);

musaError_t LaunchMusaSparseIdToMaskInt64Kernel(
    const int64_t* dense_ids, const int64_t* bound_ids,
    const int64_t* sparse_ids, float* output, int64_t output_count,
    int64_t candidate_count, int64_t sparse_count, int64_t default_id,
    musaStream_t stream);

musaError_t LaunchMusaSparseIdToMaskInt64Kernel(
    const int64_t* dense_ids, const int64_t* bound_ids,
    const int64_t* sparse_ids, int32_t* output, int64_t output_count,
    int64_t candidate_count, int64_t sparse_count, int64_t default_id,
    musaStream_t stream);
