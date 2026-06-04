#include "tensor/transpose_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void TransposeKernel(const void* input,
                                void* output,
                                int32_t element_size,
                                MusaTransposeParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.total_elements; output_index += total_threads) {
    int64_t remaining = output_index;
    int64_t input_index = 0;
    for (int32_t dim = params.rank - 1; dim >= 0; --dim) {
      const int64_t coord = remaining % params.output_dims[dim];
      remaining /= params.output_dims[dim];
      input_index += coord * params.input_strides[params.perm[dim]];
    }

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

musaError_t LaunchMusaTransposeKernel(const void* input,
                                      void* output,
                                      int32_t element_size,
                                      MusaTransposeParams params,
                                      musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  TransposeKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
      input, output, element_size, params);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}
