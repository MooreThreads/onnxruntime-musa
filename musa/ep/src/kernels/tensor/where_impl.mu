#include "tensor/where_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ int64_t ResolveWhereInputIndex(
    int64_t index,
    const MusaWhereParams& params,
    const int64_t* strides) {
  int64_t input_index = 0;
  int64_t remaining = index;
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    const int64_t coord = remaining / params.output_strides[dim];
    remaining -= coord * params.output_strides[dim];
    input_index += coord * strides[dim];
  }
  return input_index;
}

__device__ __forceinline__ void SelectElement(const void* x,
                                              const void* y,
                                              void* output,
                                              int64_t x_index,
                                              int64_t y_index,
                                              int64_t output_index,
                                              int32_t element_size,
                                              bool take_x) {
  if (element_size == 4) {
    reinterpret_cast<uint32_t*>(output)[output_index] =
        take_x ? reinterpret_cast<const uint32_t*>(x)[x_index]
               : reinterpret_cast<const uint32_t*>(y)[y_index];
  } else if (element_size == 8) {
    reinterpret_cast<uint64_t*>(output)[output_index] =
        take_x ? reinterpret_cast<const uint64_t*>(x)[x_index]
               : reinterpret_cast<const uint64_t*>(y)[y_index];
  } else if (element_size == 1) {
    reinterpret_cast<uint8_t*>(output)[output_index] =
        take_x ? reinterpret_cast<const uint8_t*>(x)[x_index]
               : reinterpret_cast<const uint8_t*>(y)[y_index];
  } else {
    const uint8_t* src =
        (take_x ? reinterpret_cast<const uint8_t*>(x) + x_index * element_size
                : reinterpret_cast<const uint8_t*>(y) + y_index * element_size);
    uint8_t* dst =
        reinterpret_cast<uint8_t*>(output) + output_index * element_size;
    for (int32_t byte = 0; byte < element_size; ++byte) {
      dst[byte] = src[byte];
    }
  }
}

__global__ void WhereKernel(const uint8_t* condition,
                            const void* x,
                            const void* y,
                            void* output,
                            int32_t element_size,
                            MusaWhereParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.total_elements;
       output_index += total_threads) {
    const int64_t condition_index =
        ResolveWhereInputIndex(output_index, params, params.condition_strides);
    const int64_t x_index =
        ResolveWhereInputIndex(output_index, params, params.x_strides);
    const int64_t y_index =
        ResolveWhereInputIndex(output_index, params, params.y_strides);
    SelectElement(x, y, output, x_index, y_index, output_index, element_size,
                  condition[condition_index] != 0);
  }
}

}  // namespace

musaError_t LaunchMusaWhereKernel(const uint8_t* condition,
                                  const void* x,
                                  const void* y,
                                  void* output,
                                  int32_t element_size,
                                  MusaWhereParams params,
                                  musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  WhereKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0,
                stream>>>(condition, x, y, output, element_size, params);
  return musaGetLastError();
}
