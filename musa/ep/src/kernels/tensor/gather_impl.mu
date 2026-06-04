#include "tensor/gather_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ int64_t ReadGatherIndex(const void* indices,
                                                   int32_t index_element_size,
                                                   int64_t offset) {
  if (index_element_size == 4) {
    return static_cast<int64_t>(reinterpret_cast<const int32_t*>(indices)[offset]);
  }
  if (index_element_size == 8) {
    return reinterpret_cast<const int64_t*>(indices)[offset];
  }
  return 0;
}

__global__ void GatherKernel(const void* input,
                             const void* indices,
                             void* output,
                             int32_t element_size,
                             int32_t index_element_size,
                             int64_t input_block_size,
                             int64_t indices_max,
                             int64_t output_block_size,
                             int64_t block_size,
                             int64_t output_count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < output_count; output_index += total_threads) {
    const int64_t input_block_index = output_index / output_block_size;
    const int64_t block_offset = output_index % output_block_size;
    const int64_t indices_index = block_offset / block_size;
    const int64_t offset = block_offset % block_size;

    int64_t gather_index = ReadGatherIndex(indices, index_element_size, indices_index);
    if (gather_index < 0) {
      gather_index += indices_max;
    }

    if (gather_index < 0 || gather_index >= indices_max) {
      if (element_size == 4) {
        reinterpret_cast<uint32_t*>(output)[output_index] = 0;
      } else if (element_size == 8) {
        reinterpret_cast<uint64_t*>(output)[output_index] = 0;
      } else if (element_size == 1) {
        reinterpret_cast<uint8_t*>(output)[output_index] = 0;
      } else {
        uint8_t* dst = reinterpret_cast<uint8_t*>(output) + output_index * element_size;
        for (int32_t byte = 0; byte < element_size; ++byte) {
          dst[byte] = 0;
        }
      }
      continue;
    }

    const int64_t input_index = input_block_index * input_block_size + gather_index * block_size + offset;
    if (element_size == 4) {
      reinterpret_cast<uint32_t*>(output)[output_index] = reinterpret_cast<const uint32_t*>(input)[input_index];
    } else if (element_size == 8) {
      reinterpret_cast<uint64_t*>(output)[output_index] = reinterpret_cast<const uint64_t*>(input)[input_index];
    } else if (element_size == 1) {
      reinterpret_cast<uint8_t*>(output)[output_index] = reinterpret_cast<const uint8_t*>(input)[input_index];
    } else {
      const uint8_t* src = reinterpret_cast<const uint8_t*>(input) + input_index * element_size;
      uint8_t* dst = reinterpret_cast<uint8_t*>(output) + output_index * element_size;
      for (int32_t byte = 0; byte < element_size; ++byte) {
        dst[byte] = src[byte];
      }
    }
  }
}

}  // namespace

musaError_t LaunchMusaGatherKernel(const void* input,
                                   const void* indices,
                                   void* output,
                                   int32_t element_size,
                                   int32_t index_element_size,
                                   int64_t input_block_size,
                                   int64_t indices_max,
                                   int64_t output_block_size,
                                   int64_t block_size,
                                   int64_t output_count,
                                   musaStream_t stream) {
  if (output_count == 0) {
    return musaSuccess;
  }
  GatherKernel<<<BlocksForCount(output_count), kThreadsPerBlock, 0, stream>>>(
      input, indices, output, element_size, index_element_size, input_block_size,
      indices_max, output_block_size, block_size, output_count);
  return musaGetLastError();
}
