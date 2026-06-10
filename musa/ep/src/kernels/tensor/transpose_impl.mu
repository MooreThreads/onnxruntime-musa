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
    } else if (element_size == 2) {
      reinterpret_cast<uint16_t*>(output)[output_index] = reinterpret_cast<const uint16_t*>(input)[input_index];
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

template <typename T>
__global__ void TransposeRank4Perm0213Kernel(const T* input,
                                             T* output,
                                             MusaTransposeParams params) {
  const int64_t d0 = params.output_dims[0];
  const int64_t d2 = params.output_dims[1];
  const int64_t d1 = params.output_dims[2];
  const int64_t d3 = params.output_dims[3];
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t output_index = thread_id; output_index < params.total_elements;
       output_index += total_threads) {
    int64_t remaining = output_index;
    const int64_t c3 = remaining % d3;
    remaining /= d3;
    const int64_t c1 = remaining % d1;
    remaining /= d1;
    const int64_t c2 = remaining % d2;
    remaining /= d2;
    const int64_t c0 = remaining;
    if (c0 >= d0) {
      continue;
    }

    const int64_t input_index = ((c0 * d1 + c1) * d2 + c2) * d3 + c3;
    output[output_index] = input[input_index];
  }
}

__global__ void TransposeRank4Perm0213Vector64Kernel(
    const uint64_t* input, uint64_t* output, int64_t total_chunks, int64_t d0,
    int64_t d1, int64_t d2, int64_t inner_chunks) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t output_index = thread_id; output_index < total_chunks;
       output_index += total_threads) {
    int64_t remaining = output_index;
    const int64_t chunk = remaining % inner_chunks;
    remaining /= inner_chunks;
    const int64_t c1 = remaining % d1;
    remaining /= d1;
    const int64_t c2 = remaining % d2;
    remaining /= d2;
    const int64_t c0 = remaining;
    if (c0 >= d0) {
      continue;
    }

    const int64_t input_index =
        ((c0 * d1 + c1) * d2 + c2) * inner_chunks + chunk;
    output[output_index] = input[input_index];
  }
}

template <typename T>
__global__ void TransposeRank4Perm0231Kernel(const T* input,
                                             T* output,
                                             MusaTransposeParams params) {
  const int64_t d0 = params.output_dims[0];
  const int64_t d2 = params.output_dims[1];
  const int64_t d3 = params.output_dims[2];
  const int64_t d1 = params.output_dims[3];
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t output_index = thread_id; output_index < params.total_elements;
       output_index += total_threads) {
    int64_t remaining = output_index;
    const int64_t c1 = remaining % d1;
    remaining /= d1;
    const int64_t c3 = remaining % d3;
    remaining /= d3;
    const int64_t c2 = remaining % d2;
    remaining /= d2;
    const int64_t c0 = remaining;
    if (c0 >= d0) {
      continue;
    }

    const int64_t input_index = ((c0 * d1 + c1) * d2 + c2) * d3 + c3;
    output[output_index] = input[input_index];
  }
}

template <typename T>
__global__ void TransposeRank4Perm0231InputLinearKernel(
    const T* input, T* output, MusaTransposeParams params) {
  const int64_t d0 = params.output_dims[0];
  const int64_t d2 = params.output_dims[1];
  const int64_t d3 = params.output_dims[2];
  const int64_t d1 = params.output_dims[3];
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t input_index = thread_id; input_index < params.total_elements;
       input_index += total_threads) {
    int64_t remaining = input_index;
    const int64_t c3 = remaining % d3;
    remaining /= d3;
    const int64_t c2 = remaining % d2;
    remaining /= d2;
    const int64_t c1 = remaining % d1;
    remaining /= d1;
    const int64_t c0 = remaining;
    if (c0 >= d0) {
      continue;
    }

    const int64_t output_index = ((c0 * d2 + c2) * d3 + c3) * d1 + c1;
    output[output_index] = input[input_index];
  }
}

template <typename T, int TileDim, int BlockRows>
__global__ void TransposeRank4Perm0231TiledKernel(const T* input,
                                                  T* output,
                                                  int64_t d0,
                                                  int64_t d1,
                                                  int64_t flattened_inner) {
  __shared__ T tile[TileDim][TileDim + 1];

  const int64_t batch = static_cast<int64_t>(blockIdx.z);
  const int64_t input_row =
      static_cast<int64_t>(blockIdx.y) * TileDim + threadIdx.y;
  const int64_t input_col =
      static_cast<int64_t>(blockIdx.x) * TileDim + threadIdx.x;

  if (batch < d0 && input_row < d1 && input_col < flattened_inner) {
    const int64_t input_index =
        (batch * d1 + input_row) * flattened_inner + input_col;
    tile[threadIdx.y][threadIdx.x] = input[input_index];
  }
  __syncthreads();

  const int64_t output_row =
      static_cast<int64_t>(blockIdx.x) * TileDim + threadIdx.y;
  const int64_t output_col =
      static_cast<int64_t>(blockIdx.y) * TileDim + threadIdx.x;
  if (batch < d0 && output_row < flattened_inner && output_col < d1) {
    const int64_t output_index =
        (batch * flattened_inner + output_row) * d1 + output_col;
    output[output_index] = tile[threadIdx.x][threadIdx.y];
  }
}

template <typename T>
musaError_t LaunchRank4TransposeTyped(const void* input, void* output,
                                      MusaTransposeParams params,
                                      musaStream_t stream) {
  const bool perm_0213 = params.perm[0] == 0 && params.perm[1] == 2 &&
                         params.perm[2] == 1 && params.perm[3] == 3;
  const bool perm_0231 = params.perm[0] == 0 && params.perm[1] == 2 &&
                         params.perm[2] == 3 && params.perm[3] == 1;
  if (perm_0213) {
    TransposeRank4Perm0213Kernel<T>
        <<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0,
           stream>>>(reinterpret_cast<const T*>(input),
                     reinterpret_cast<T*>(output), params);
    return musaGetLastError();
  }
  if (perm_0231) {
    constexpr int kTileDim = 16;
    constexpr int kBlockRows = 16;
    const int64_t d0 = params.output_dims[0];
    const int64_t d1 = params.output_dims[3];
    const int64_t flattened_inner =
        params.output_dims[1] * params.output_dims[2];
    const dim3 block(kTileDim, kBlockRows, 1);
    const dim3 grid(
        static_cast<unsigned int>((flattened_inner + kTileDim - 1) / kTileDim),
        static_cast<unsigned int>((d1 + kTileDim - 1) / kTileDim),
        static_cast<unsigned int>(d0));
    TransposeRank4Perm0231TiledKernel<T, kTileDim, kBlockRows>
        <<<grid, block, 0, stream>>>(reinterpret_cast<const T*>(input),
                                     reinterpret_cast<T*>(output), d0, d1,
                                     flattened_inner);
    return musaGetLastError();
  }
  return musaErrorNotSupported;
}

musaError_t LaunchRank4Transpose0213Vector64(const void* input, void* output,
                                             int32_t element_size,
                                             MusaTransposeParams params,
                                             musaStream_t stream) {
  if (!(params.perm[0] == 0 && params.perm[1] == 2 && params.perm[2] == 1 &&
        params.perm[3] == 3)) {
    return musaErrorNotSupported;
  }
  const int64_t inner_bytes = params.output_dims[3] * element_size;
  if (inner_bytes <= 0 ||
      inner_bytes % static_cast<int64_t>(sizeof(uint64_t)) != 0) {
    return musaErrorNotSupported;
  }
  const int64_t inner_chunks =
      inner_bytes / static_cast<int64_t>(sizeof(uint64_t));
  const int64_t total_chunks =
      params.output_dims[0] * params.output_dims[1] * params.output_dims[2] *
      inner_chunks;
  TransposeRank4Perm0213Vector64Kernel
      <<<BlocksForCount(total_chunks), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const uint64_t*>(input),
          reinterpret_cast<uint64_t*>(output), total_chunks,
          params.output_dims[0], params.output_dims[2], params.output_dims[1],
          inner_chunks);
  return musaGetLastError();
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
  if (params.rank == 4) {
    musaError_t status = musaErrorNotSupported;
    status = LaunchRank4Transpose0213Vector64(input, output, element_size,
                                              params, stream);
    if (status != musaErrorNotSupported) {
      return status;
    }
    if (element_size == 1) {
      status =
          LaunchRank4TransposeTyped<uint8_t>(input, output, params, stream);
    } else if (element_size == 2) {
      status =
          LaunchRank4TransposeTyped<uint16_t>(input, output, params, stream);
    } else if (element_size == 4) {
      status =
          LaunchRank4TransposeTyped<uint32_t>(input, output, params, stream);
    } else if (element_size == 8) {
      status =
          LaunchRank4TransposeTyped<uint64_t>(input, output, params, stream);
    }
    if (status != musaErrorNotSupported) {
      return status;
    }
  }
  TransposeKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
      input, output, element_size, params);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}
