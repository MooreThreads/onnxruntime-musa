#include "tensor/tile_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

#include <stdint.h>

namespace {

__device__ __forceinline__ void CopyElement(const void* input,
                                            void* output,
                                            int64_t input_index,
                                            int64_t output_index,
                                            int32_t element_size) {
  if (element_size == 4) {
    reinterpret_cast<uint32_t*>(output)[output_index] =
        reinterpret_cast<const uint32_t*>(input)[input_index];
  } else if (element_size == 8) {
    reinterpret_cast<uint64_t*>(output)[output_index] =
        reinterpret_cast<const uint64_t*>(input)[input_index];
  } else if (element_size == 1) {
    reinterpret_cast<uint8_t*>(output)[output_index] =
        reinterpret_cast<const uint8_t*>(input)[input_index];
  } else {
    const uint8_t* src =
        reinterpret_cast<const uint8_t*>(input) + input_index * element_size;
    uint8_t* dst =
        reinterpret_cast<uint8_t*>(output) + output_index * element_size;
    for (int32_t byte = 0; byte < element_size; ++byte) {
      dst[byte] = src[byte];
    }
  }
}

__global__ void TileKernel(const void* input,
                           void* output,
                           int32_t element_size,
                           MusaTileParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.total_elements;
       output_index += total_threads) {
    int64_t remaining = output_index;
    int64_t input_index = 0;
    for (int32_t dim = params.rank - 1; dim >= 0; --dim) {
      const int64_t coord = remaining % params.output_dims[dim];
      remaining /= params.output_dims[dim];
      const int64_t input_dim = params.input_dims[dim];
      const int64_t input_coord = input_dim == 0 ? 0 : coord % input_dim;
      input_index += input_coord * params.input_strides[dim];
    }
    CopyElement(input, output, input_index, output_index, element_size);
  }
}

template <typename T>
__global__ void TileLastDimKernel(const T* input,
                                  T* output,
                                  int64_t rows,
                                  int64_t cols,
                                  int64_t repeats) {
  const int64_t input_elements = rows * cols;
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t input_index = thread_id; input_index < input_elements;
       input_index += total_threads) {
    const T value = input[input_index];
    const int64_t row = input_index / cols;
    const int64_t col = input_index - row * cols;
    int64_t output_index = row * cols * repeats + col;
    for (int64_t repeat = 0; repeat < repeats; ++repeat) {
      output[output_index] = value;
      output_index += cols;
    }
  }
}

template <typename T>
__global__ void TileLastDimBroadcastKernel(const T* input,
                                           T* output,
                                           int64_t rows,
                                           int64_t repeats) {
  const int64_t output_elements = rows * repeats;
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < output_elements;
       output_index += total_threads) {
    output[output_index] = input[output_index / repeats];
  }
}

}  // namespace

musaError_t LaunchMusaTileKernel(const void* input,
                                 void* output,
                                 int32_t element_size,
                                 MusaTileParams params,
                                 musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  TileKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0,
               stream>>>(input, output, element_size, params);
  return musaGetLastError();
}

musaError_t LaunchMusaTileLastDimKernel(const void* input,
                                        void* output,
                                        int32_t element_size,
                                        int64_t rows,
                                        int64_t cols,
                                        int64_t repeats,
                                        musaStream_t stream) {
  if (rows == 0 || cols == 0 || repeats == 0) {
    return musaSuccess;
  }
  const int64_t input_elements = rows * cols;
  const int64_t output_elements = rows * cols * repeats;
  switch (element_size) {
    case 1:
      if (cols == 1) {
        TileLastDimBroadcastKernel<uint8_t>
            <<<BlocksForCount(output_elements), kThreadsPerBlock, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(input),
                reinterpret_cast<uint8_t*>(output), rows, repeats);
      } else {
        TileLastDimKernel<uint8_t><<<BlocksForCount(input_elements),
                                     kThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(input),
            reinterpret_cast<uint8_t*>(output), rows, cols, repeats);
      }
      break;
    case 2:
      if (cols == 1) {
        TileLastDimBroadcastKernel<uint16_t>
            <<<BlocksForCount(output_elements), kThreadsPerBlock, 0, stream>>>(
                reinterpret_cast<const uint16_t*>(input),
                reinterpret_cast<uint16_t*>(output), rows, repeats);
      } else {
        TileLastDimKernel<uint16_t><<<BlocksForCount(input_elements),
                                      kThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(input),
            reinterpret_cast<uint16_t*>(output), rows, cols, repeats);
      }
      break;
    case 4:
      if (cols == 1) {
        TileLastDimBroadcastKernel<uint32_t>
            <<<BlocksForCount(output_elements), kThreadsPerBlock, 0, stream>>>(
                reinterpret_cast<const uint32_t*>(input),
                reinterpret_cast<uint32_t*>(output), rows, repeats);
      } else {
        TileLastDimKernel<uint32_t><<<BlocksForCount(input_elements),
                                      kThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(input),
            reinterpret_cast<uint32_t*>(output), rows, cols, repeats);
      }
      break;
    case 8:
      if (cols == 1) {
        TileLastDimBroadcastKernel<uint64_t>
            <<<BlocksForCount(output_elements), kThreadsPerBlock, 0, stream>>>(
                reinterpret_cast<const uint64_t*>(input),
                reinterpret_cast<uint64_t*>(output), rows, repeats);
      } else {
        TileLastDimKernel<uint64_t><<<BlocksForCount(input_elements),
                                      kThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const uint64_t*>(input),
            reinterpret_cast<uint64_t*>(output), rows, cols, repeats);
      }
      break;
    default:
      return musaErrorNotSupported;
  }
  return musaGetLastError();
}
