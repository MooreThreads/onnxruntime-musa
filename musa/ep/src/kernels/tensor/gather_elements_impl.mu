#include "shared_inc/musa_kernel_common.mu.h"
#include "tensor/gather_elements_impl.h"

namespace {

__device__ __forceinline__ void CopyGatherElement(const void* input,
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

__device__ __forceinline__ int64_t ReadGatherIndex(const void* indices,
                                                   int64_t offset,
                                                   int32_t index_element_size) {
  if (index_element_size == 4) {
    return static_cast<int64_t>(reinterpret_cast<const int32_t*>(indices)[offset]);
  }
  return reinterpret_cast<const int64_t*>(indices)[offset];
}

__global__ void GatherElementsKernel(const void* data, const void* indices,
                                     void* output, int32_t element_size,
                                     int32_t index_element_size,
                                     MusaGatherElementsParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t output_index = thread_id; output_index < params.output_elements;
       output_index += total_threads) {
    int64_t remaining = output_index;
    int64_t data_offset = 0;
    for (int32_t dim = 0; dim < params.rank; ++dim) {
      const int64_t coord = remaining / params.indices_strides[dim];
      remaining -= coord * params.indices_strides[dim];
      if (dim == params.axis) {
        int64_t gather_index =
            ReadGatherIndex(indices, output_index, index_element_size);
        const int64_t dim_size = params.data_dims[dim];
        if (gather_index < 0) {
          gather_index += dim_size;
        }
        if (gather_index < 0 || gather_index >= dim_size) {
          auto* dst =
              reinterpret_cast<uint8_t*>(output) + output_index * element_size;
          for (int32_t byte = 0; byte < element_size; ++byte) {
            dst[byte] = 0;
          }
          data_offset = -1;
          break;
        }
        data_offset += gather_index * params.data_strides[dim];
      } else {
        data_offset += coord * params.data_strides[dim];
      }
    }
    if (data_offset < 0) {
      continue;
    }
    CopyGatherElement(data, output, data_offset, output_index, element_size);
  }
}

}  // namespace

musaError_t LaunchMusaGatherElementsKernel(
    const void* data, const void* indices, void* output, int32_t element_size,
    int32_t index_element_size, MusaGatherElementsParams params,
    musaStream_t stream) {
  if (params.output_elements == 0) {
    return musaSuccess;
  }
  GatherElementsKernel<<<BlocksForCount(params.output_elements),
                         kThreadsPerBlock, 0, stream>>>(
      data, indices, output, element_size, index_element_size, params);
  return musaGetLastError();
}
