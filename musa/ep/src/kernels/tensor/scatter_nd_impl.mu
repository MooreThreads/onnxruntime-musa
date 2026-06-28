#include "shared_inc/musa_kernel_common.mu.h"
#include "tensor/scatter_nd_impl.h"

namespace {
constexpr int32_t kScatterNDReductionAdd = 1;

__device__ __forceinline__ void AtomicAddScatterNDElement(
    void* output, const void* updates, int64_t output_index,
    int64_t updates_index, int32_t element_size) {
  if (element_size == 4) {
    atomicAdd(reinterpret_cast<int32_t*>(output) + output_index,
              reinterpret_cast<const int32_t*>(updates)[updates_index]);
  } else if (element_size == 8) {
    atomicAdd(reinterpret_cast<unsigned long long*>(output) + output_index,
              reinterpret_cast<const unsigned long long*>(
                  updates)[updates_index]);
  }
}

__device__ __forceinline__ void CopyScatterNDElement(void* output,
                                                     const void* updates,
                                                     int64_t output_index,
                                                     int64_t updates_index,
                                                     int32_t element_size) {
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
  } else {
    auto* dst =
        reinterpret_cast<uint8_t*>(output) + output_index * element_size;
    const auto* src =
        reinterpret_cast<const uint8_t*>(updates) + updates_index * element_size;
    for (int32_t byte = 0; byte < element_size; ++byte) {
      dst[byte] = src[byte];
    }
  }
}

__global__ void ScatterNDKernel(void* output, const int64_t* indices,
                                const void* updates, int32_t element_size,
                                MusaScatterNDParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t total_updates =
      params.num_indices * params.updates_slice_size;

  for (int64_t updates_index = thread_id; updates_index < total_updates;
       updates_index += total_threads) {
    const int64_t indices_row =
        params.updates_slice_size == 0
            ? 0
            : updates_index / params.updates_slice_size;
    const int64_t element_offset =
        params.updates_slice_size == 0
            ? 0
            : updates_index % params.updates_slice_size;

    int64_t output_offset = 0;
    const int64_t indices_base =
        indices_row * params.last_index_dimension;
    for (int32_t dim = 0; dim < params.last_index_dimension; ++dim) {
      int64_t index = indices[indices_base + dim];
      const int64_t dim_size = params.input_dims[dim];
      if (index >= 0) {
        if (index >= dim_size) {
          index = dim_size - 1;
        }
      } else if (index < -dim_size) {
        index = 0;
      } else {
        index += dim_size;
      }
      output_offset += index * params.input_strides[dim];
    }

    if (params.reduction == kScatterNDReductionAdd) {
      AtomicAddScatterNDElement(output, updates, output_offset + element_offset,
                                updates_index, element_size);
    } else {
      CopyScatterNDElement(output, updates, output_offset + element_offset,
                           updates_index, element_size);
    }
  }
}

}  // namespace

musaError_t LaunchMusaScatterNDKernel(void* output, const int64_t* indices,
                                      const void* updates,
                                      int32_t element_size,
                                      MusaScatterNDParams params,
                                      musaStream_t stream) {
  const int64_t total_updates =
      params.num_indices * params.updates_slice_size;
  if (total_updates == 0) {
    return musaSuccess;
  }
  ScatterNDKernel<<<BlocksForCount(total_updates), kThreadsPerBlock, 0,
                    stream>>>(output, indices, updates, element_size, params);
  return musaGetLastError();
}
