#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaMaskedEmbeddingLookupKernel(
    const void* table, const void* ids, void* output, int32_t element_size,
    int32_t index_element_size, int64_t sequence_count, int64_t table_rows,
    int64_t embedding_dim, int64_t threshold, musaStream_t stream);
