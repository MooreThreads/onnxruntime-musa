#pragma once

#include <musa_runtime.h>

#include <cstdint>

musaError_t LaunchMusaTargetIdCountEmbeddingInt32Kernel(
    const int32_t* ids, const void* target, const float* table,
    float* embedding_output, float* count_output, int64_t rows,
    int64_t sequence_length, int64_t embedding_dim, int64_t table_rows,
    int64_t target_count, int64_t target_elem_size, int64_t target_constant,
    int64_t pad, int64_t cap, musaStream_t stream);

musaError_t LaunchMusaTargetIdCountEmbeddingInt64Kernel(
    const int64_t* ids, const void* target, const float* table,
    float* embedding_output, float* count_output, int64_t rows,
    int64_t sequence_length, int64_t embedding_dim, int64_t table_rows,
    int64_t target_count, int64_t target_elem_size, int64_t target_constant,
    int64_t pad, int64_t cap, musaStream_t stream);
