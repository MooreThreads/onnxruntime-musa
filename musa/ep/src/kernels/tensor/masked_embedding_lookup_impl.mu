#include "tensor/masked_embedding_lookup_impl.h"

#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ int64_t ReadId(const void* ids,
                                          int32_t index_element_size,
                                          int64_t offset) {
  if (index_element_size == 4) {
    return static_cast<int64_t>(reinterpret_cast<const int32_t*>(ids)[offset]);
  }
  if (index_element_size == 8) {
    return reinterpret_cast<const int64_t*>(ids)[offset];
  }
  return 0;
}

__device__ __forceinline__ void CopyElement(const void* table, void* output,
                                            int64_t table_index,
                                            int64_t output_index,
                                            int32_t element_size) {
  if (element_size == 4) {
    reinterpret_cast<uint32_t*>(output)[output_index] =
        reinterpret_cast<const uint32_t*>(table)[table_index];
  } else if (element_size == 8) {
    reinterpret_cast<uint64_t*>(output)[output_index] =
        reinterpret_cast<const uint64_t*>(table)[table_index];
  } else if (element_size == 2) {
    reinterpret_cast<uint16_t*>(output)[output_index] =
        reinterpret_cast<const uint16_t*>(table)[table_index];
  } else if (element_size == 1) {
    reinterpret_cast<uint8_t*>(output)[output_index] =
        reinterpret_cast<const uint8_t*>(table)[table_index];
  } else {
    const auto* src =
        reinterpret_cast<const uint8_t*>(table) + table_index * element_size;
    auto* dst =
        reinterpret_cast<uint8_t*>(output) + output_index * element_size;
    for (int32_t byte = 0; byte < element_size; ++byte) {
      dst[byte] = src[byte];
    }
  }
}

__global__ void MaskedEmbeddingLookupKernel(
    const void* table, const void* ids, void* output, int32_t element_size,
    int32_t index_element_size, int64_t sequence_count, int64_t table_rows,
    int64_t embedding_dim, int64_t threshold, int64_t output_count) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t output_index = thread_id; output_index < output_count;
       output_index += total_threads) {
    const int64_t sequence_index = output_index / embedding_dim;
    const int64_t embedding_index = output_index % embedding_dim;
    const int64_t id = ReadId(ids, index_element_size, sequence_index);
    if (id < threshold || id < 0 || id >= table_rows) {
      continue;
    }

    CopyElement(table, output, id * embedding_dim + embedding_index,
                output_index, element_size);
  }
}

}  // namespace

musaError_t LaunchMusaMaskedEmbeddingLookupKernel(
    const void* table, const void* ids, void* output, int32_t element_size,
    int32_t index_element_size, int64_t sequence_count, int64_t table_rows,
    int64_t embedding_dim, int64_t threshold, musaStream_t stream) {
  const int64_t output_count = sequence_count * embedding_dim;
  if (output_count == 0) {
    return musaSuccess;
  }
  MaskedEmbeddingLookupKernel<<<BlocksForCount(output_count), kThreadsPerBlock,
                                0, stream>>>(
      table, ids, output, element_size, index_element_size, sequence_count,
      table_rows, embedding_dim, threshold, output_count);
  return musaGetLastError();
}
