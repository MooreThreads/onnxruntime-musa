#include "shared_inc/musa_kernel_common.mu.h"
#include "tensor/scatter_elements_impl.h"

namespace {

constexpr int32_t kScatterElementsReductionNone = 0;
constexpr int32_t kScatterElementsReductionAdd = 1;

__device__ __forceinline__ int64_t ReadScatterElementIndex(
    const void* indices, int64_t offset, int32_t index_element_size) {
  if (index_element_size == 4) {
    return static_cast<int64_t>(
        reinterpret_cast<const int32_t*>(indices)[offset]);
  }
  return reinterpret_cast<const int64_t*>(indices)[offset];
}

__device__ __forceinline__ void WriteScatterElement(
    void* output, const void* updates, int64_t output_index,
    int64_t updates_index, int32_t element_size) {
  if (element_size == 8) {
    reinterpret_cast<uint64_t*>(output)[output_index] =
        reinterpret_cast<const uint64_t*>(updates)[updates_index];
  } else if (element_size == 4) {
    reinterpret_cast<uint32_t*>(output)[output_index] =
        reinterpret_cast<const uint32_t*>(updates)[updates_index];
  } else if (element_size == 2) {
    reinterpret_cast<uint16_t*>(output)[output_index] =
        reinterpret_cast<const uint16_t*>(updates)[updates_index];
  } else if (element_size == 1) {
    reinterpret_cast<uint8_t*>(output)[output_index] =
        reinterpret_cast<const uint8_t*>(updates)[updates_index];
  }
}

__device__ __forceinline__ void AtomicAddScatterElement(
    void* output, const void* updates, int64_t output_index,
    int64_t updates_index, int32_t elem_type) {
  constexpr int32_t kFloat = 1;
  constexpr int32_t kInt32 = 6;
  constexpr int32_t kInt64 = 7;
  constexpr int32_t kDouble = 11;
  if (elem_type == kFloat) {
    atomicAdd(reinterpret_cast<float*>(output) + output_index,
              reinterpret_cast<const float*>(updates)[updates_index]);
  } else if (elem_type == kDouble) {
    atomicAdd(reinterpret_cast<double*>(output) + output_index,
              reinterpret_cast<const double*>(updates)[updates_index]);
  } else if (elem_type == kInt32) {
    atomicAdd(reinterpret_cast<int32_t*>(output) + output_index,
              reinterpret_cast<const int32_t*>(updates)[updates_index]);
  } else if (elem_type == kInt64) {
    atomicAdd(reinterpret_cast<unsigned long long*>(output) + output_index,
              static_cast<unsigned long long>(
                  reinterpret_cast<const int64_t*>(updates)[updates_index]));
  }
}

__global__ void ScatterElementsKernel(
    void* output, const void* indices, const void* updates,
    int32_t element_size, int32_t index_element_size, int32_t elem_type,
    MusaScatterElementsParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t updates_index = thread_id;
       updates_index < params.updates_elements;
       updates_index += total_threads) {
    int64_t remaining = updates_index;
    int64_t output_offset = 0;
    for (int32_t dim = 0; dim < params.rank; ++dim) {
      int64_t coord = remaining / params.updates_strides[dim];
      remaining -= coord * params.updates_strides[dim];
      if (dim == params.axis) {
        coord = ReadScatterElementIndex(indices, updates_index,
                                        index_element_size);
        const int64_t dim_size = params.data_dims[dim];
        if (coord < 0) {
          coord += dim_size;
        }
        if (coord < 0 || coord >= dim_size) {
          continue;
        }
      }
      output_offset += coord * params.data_strides[dim];
    }

    if (params.reduction == kScatterElementsReductionAdd) {
      AtomicAddScatterElement(output, updates, output_offset, updates_index,
                              elem_type);
    } else {
      WriteScatterElement(output, updates, output_offset, updates_index,
                          element_size);
    }
  }
}

}  // namespace

musaError_t LaunchMusaScatterElementsKernel(
    void* output, const void* indices, const void* updates,
    int32_t element_size, int32_t index_element_size, int32_t elem_type,
    MusaScatterElementsParams params, musaStream_t stream) {
  if (params.updates_elements == 0) {
    return musaSuccess;
  }
  ScatterElementsKernel<<<BlocksForCount(params.updates_elements),
                          kThreadsPerBlock, 0, stream>>>(
      output, indices, updates, element_size, index_element_size, elem_type,
      params);
  return musaGetLastError();
}
