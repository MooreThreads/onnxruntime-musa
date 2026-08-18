#pragma once

#include <musa_runtime.h>

#include <cstdint>

musaError_t LaunchMusaLogBucketizeGatherKernel(
    const void* table, const void* current, const void* sequence, void* output,
    int32_t element_size, int32_t current_element_size,
    int32_t sequence_element_size, int64_t current_count,
    int64_t sequence_count, int64_t table_rows, int64_t block_size,
    float diff_min, float time_scale, float value_min, float log_base,
    float bucket_start, float bucket_span, float bucket_scale,
    int64_t bucket_offset, int64_t bucket_min, int64_t bucket_max,
    float lower_threshold, float upper_threshold, musaStream_t stream);
