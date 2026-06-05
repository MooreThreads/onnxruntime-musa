#include "shared_inc/musa_kernel_common.mu.h"
#include "tensor/gather_nd_impl.h"

namespace {

__device__ __forceinline__ void CopyGatherNDElement(const void* input,
                                                    void* output,
                                                    int64_t input_index,
                                                    int64_t output_index,
                                                    int32_t element_size) {
  if (element_size == 8) {
    reinterpret_cast<uint64_t*>(output)[output_index] =
        reinterpret_cast<const uint64_t*>(input)[input_index];
  } else if (element_size == 4) {
    reinterpret_cast<uint32_t*>(output)[output_index] =
        reinterpret_cast<const uint32_t*>(input)[input_index];
  } else if (element_size == 2) {
    reinterpret_cast<uint16_t*>(output)[output_index] =
        reinterpret_cast<const uint16_t*>(input)[input_index];
  } else if (element_size == 1) {
    reinterpret_cast<uint8_t*>(output)[output_index] =
        reinterpret_cast<const uint8_t*>(input)[input_index];
  } else {
    const auto* src =
        reinterpret_cast<const uint8_t*>(input) + input_index * element_size;
    auto* dst =
        reinterpret_cast<uint8_t*>(output) + output_index * element_size;
    for (int32_t byte = 0; byte < element_size; ++byte) {
      dst[byte] = src[byte];
    }
  }
}

__global__ void GatherNDKernel(const void* input, const int64_t* indices,
                               void* output, int32_t element_size,
                               MusaGatherNDParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.output_elements;
       output_index += total_threads) {
    const int64_t slice_index =
        params.slice_size == 0 ? 0 : output_index / params.slice_size;
    const int64_t element_offset =
        params.slice_size == 0 ? 0 : output_index % params.slice_size;
    const int64_t batch_index = params.num_slices_per_batch == 0
                                    ? 0
                                    : slice_index / params.num_slices_per_batch;

    int64_t input_offset = batch_index * params.input_batch_stride;
    const int64_t indices_base = slice_index * params.num_slice_dims;
    bool valid = true;
    for (int32_t dim = 0; dim < params.num_slice_dims; ++dim) {
      const int32_t input_dim = params.batch_dims + dim;
      int64_t index = indices[indices_base + dim];
      const int64_t dim_size = params.input_dims[input_dim];
      if (index < 0) {
        index += dim_size;
      }
      if (index < 0 || index >= dim_size) {
        valid = false;
        break;
      }
      input_offset += index * params.sizes_from_slice_dims[dim];
    }

    if (valid) {
      CopyGatherNDElement(input, output, input_offset + element_offset,
                          output_index, element_size);
    } else {
      auto* dst =
          reinterpret_cast<uint8_t*>(output) + output_index * element_size;
      for (int32_t byte = 0; byte < element_size; ++byte) {
        dst[byte] = 0;
      }
    }
  }
}

}  // namespace

musaError_t LaunchMusaGatherNDKernel(const void* input, const int64_t* indices,
                                     void* output, int32_t element_size,
                                     MusaGatherNDParams params,
                                     musaStream_t stream) {
  if (params.output_elements == 0) {
    return musaSuccess;
  }
  GatherNDKernel<<<BlocksForCount(params.output_elements), kThreadsPerBlock, 0,
                   stream>>>(input, indices, output, element_size, params);
  return musaGetLastError();
}
