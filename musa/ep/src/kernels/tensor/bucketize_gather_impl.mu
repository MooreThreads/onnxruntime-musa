#include "shared_inc/musa_kernel_common.mu.h"
#include "tensor/bucketize_gather_impl.h"

namespace {

__device__ __forceinline__ int64_t ReadIndex(const void* indices,
                                             int32_t index_element_size,
                                             int64_t offset) {
  if (index_element_size == 4) {
    return static_cast<int64_t>(
        reinterpret_cast<const int32_t*>(indices)[offset]);
  }
  if (index_element_size == 8) {
    return reinterpret_cast<const int64_t*>(indices)[offset];
  }
  return 0;
}

__global__ void BucketizeGatherKernel(const void* table, const void* indices,
                                      void* output, int32_t element_size,
                                      int32_t index_element_size,
                                      int64_t indices_count, int64_t table_rows,
                                      int64_t block_size, int64_t modulus,
                                      int64_t offset, float greater_threshold,
                                      int64_t output_count) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < output_count;
       output_index += total_threads) {
    const int64_t indices_index = output_index / block_size;
    const int64_t element_offset = output_index % block_size;
    const int64_t raw_index =
        ReadIndex(indices, index_element_size, indices_index);
    const int64_t remainder = raw_index - (raw_index / modulus) * modulus;
    int64_t gather_index = static_cast<float>(raw_index) > greater_threshold
                               ? remainder + offset
                               : 0;
    if (gather_index < 0 || gather_index >= table_rows) {
      gather_index = 0;
    }

    const int64_t table_index = gather_index * block_size + element_offset;
    if (element_size == 4) {
      reinterpret_cast<uint32_t*>(output)[output_index] =
          reinterpret_cast<const uint32_t*>(table)[table_index];
    } else if (element_size == 8) {
      reinterpret_cast<uint64_t*>(output)[output_index] =
          reinterpret_cast<const uint64_t*>(table)[table_index];
    } else if (element_size == 1) {
      reinterpret_cast<uint8_t*>(output)[output_index] =
          reinterpret_cast<const uint8_t*>(table)[table_index];
    } else {
      const uint8_t* src =
          reinterpret_cast<const uint8_t*>(table) + table_index * element_size;
      uint8_t* dst =
          reinterpret_cast<uint8_t*>(output) + output_index * element_size;
      for (int32_t byte = 0; byte < element_size; ++byte) {
        dst[byte] = src[byte];
      }
    }
  }
}

}  // namespace

musaError_t LaunchMusaBucketizeGatherKernel(
    const void* table, const void* indices, void* output, int32_t element_size,
    int32_t index_element_size, int64_t indices_count, int64_t table_rows,
    int64_t block_size, int64_t modulus, int64_t offset,
    float greater_threshold, musaStream_t stream) {
  const int64_t output_count = indices_count * block_size;
  if (output_count == 0) {
    return musaSuccess;
  }
  BucketizeGatherKernel<<<BlocksForCount(output_count), kThreadsPerBlock, 0,
                          stream>>>(
      table, indices, output, element_size, index_element_size, indices_count,
      table_rows, block_size, modulus, offset, greater_threshold, output_count);
  return musaGetLastError();
}
