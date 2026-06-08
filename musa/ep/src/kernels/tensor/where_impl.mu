#include "tensor/where_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

constexpr int kWhereItemsPerThread = 4;

__host__ __forceinline__ bool IsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

int BlocksForWhereCount(int64_t count, int64_t items_per_thread = 1) {
  return BlocksForCount((count + items_per_thread - 1) / items_per_thread);
}

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

template <typename T>
__global__ void WhereSameShapeFastKernel(const uint8_t* condition,
                                         const T* x,
                                         const T* y,
                                         T* output,
                                         int64_t total_elements) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kWhereItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x * kWhereItemsPerThread;
  for (; index < total_elements; index += stride) {
#pragma unroll
    for (int item = 0; item < kWhereItemsPerThread; ++item) {
      const int64_t offset = index + item;
      if (offset < total_elements) {
        output[offset] = condition[offset] != 0 ? x[offset] : y[offset];
      }
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void WhereSameShapeFloat4Kernel(
    const uint8_t* condition,
    const float4* x,
    const float4* y,
    float4* output,
    int64_t vec_elements) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; index < vec_elements; index += stride) {
    const int64_t base = index * 4;
    const float4 xv = x[index];
    const float4 yv = y[index];
    output[index] = make_float4(condition[base] != 0 ? xv.x : yv.x,
                                condition[base + 1] != 0 ? xv.y : yv.y,
                                condition[base + 2] != 0 ? xv.z : yv.z,
                                condition[base + 3] != 0 ? xv.w : yv.w);
  }
}

template <typename T>
__global__ void WhereRowwiseFastKernel(const uint8_t* condition,
                                       const T* x,
                                       const T* y,
                                       T* output,
                                       int64_t total_elements,
                                       int64_t inner_size,
                                       bool x_broadcast_rows,
                                       bool y_broadcast_rows) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kWhereItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x * kWhereItemsPerThread;
  for (; index < total_elements; index += stride) {
#pragma unroll
    for (int item = 0; item < kWhereItemsPerThread; ++item) {
      const int64_t offset = index + item;
      if (offset < total_elements) {
        const int64_t row = offset / inner_size;
        const int64_t col = offset - row * inner_size;
        const int64_t x_index = x_broadcast_rows ? col : offset;
        const int64_t y_index = y_broadcast_rows ? col : offset;
        output[offset] = condition[row] != 0 ? x[x_index] : y[y_index];
      }
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void WhereRowwiseFloat4Kernel(
    const uint8_t* condition,
    const float4* x,
    const float4* y,
    float4* output,
    int64_t rows,
    int64_t inner_vecs,
    bool x_broadcast_rows,
    bool y_broadcast_rows) {
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const bool take_x = condition[row] != 0;
    const int64_t row_offset = row * inner_vecs;
    int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; col < inner_vecs; col += stride) {
      const int64_t output_index = row_offset + col;
      const int64_t x_index = x_broadcast_rows ? col : output_index;
      const int64_t y_index = y_broadcast_rows ? col : output_index;
      output[output_index] = take_x ? x[x_index] : y[y_index];
    }
  }
}

template <typename T>
musaError_t LaunchWhereSameShapeTyped(const uint8_t* condition,
                                      const void* x,
                                      const void* y,
                                      void* output,
                                      int64_t total_elements,
                                      musaStream_t stream) {
  WhereSameShapeFastKernel<T>
      <<<BlocksForWhereCount(total_elements, kWhereItemsPerThread),
         kThreadsPerBlock, 0, stream>>>(
          condition, reinterpret_cast<const T*>(x),
          reinterpret_cast<const T*>(y), reinterpret_cast<T*>(output),
          total_elements);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchWhereRowwiseTyped(const uint8_t* condition,
                                    const void* x,
                                    const void* y,
                                    void* output,
                                    int64_t rows,
                                    int64_t inner_size,
                                    bool x_broadcast_rows,
                                    bool y_broadcast_rows,
                                    musaStream_t stream) {
  const int64_t total_elements = rows * inner_size;
  WhereRowwiseFastKernel<T>
      <<<BlocksForWhereCount(total_elements, kWhereItemsPerThread),
         kThreadsPerBlock, 0, stream>>>(
          condition, reinterpret_cast<const T*>(x),
          reinterpret_cast<const T*>(y), reinterpret_cast<T*>(output),
          total_elements, inner_size, x_broadcast_rows, y_broadcast_rows);
  return musaGetLastError();
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

musaError_t LaunchMusaWhereSameShapeFastKernel(const uint8_t* condition,
                                               const void* x,
                                               const void* y,
                                               void* output,
                                               int32_t element_size,
                                               int64_t total_elements,
                                               musaStream_t stream) {
  if (total_elements == 0) {
    return musaSuccess;
  }
  if (element_size == 4 && total_elements % 4 == 0 && IsAligned16(x) &&
      IsAligned16(y) && IsAligned16(output)) {
    const int64_t vec_elements = total_elements / 4;
    WhereSameShapeFloat4Kernel<<<BlocksForCount(vec_elements), kThreadsPerBlock,
                                 0, stream>>>(
        condition, reinterpret_cast<const float4*>(x),
        reinterpret_cast<const float4*>(y), reinterpret_cast<float4*>(output),
        vec_elements);
    return musaGetLastError();
  }

  switch (element_size) {
    case 1:
      return LaunchWhereSameShapeTyped<uint8_t>(condition, x, y, output,
                                                total_elements, stream);
    case 2:
      return LaunchWhereSameShapeTyped<uint16_t>(condition, x, y, output,
                                                 total_elements, stream);
    case 4:
      return LaunchWhereSameShapeTyped<uint32_t>(condition, x, y, output,
                                                 total_elements, stream);
    case 8:
      return LaunchWhereSameShapeTyped<uint64_t>(condition, x, y, output,
                                                 total_elements, stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaWhereRowwiseFastKernel(const uint8_t* condition,
                                             const void* x,
                                             const void* y,
                                             void* output,
                                             int32_t element_size,
                                             int64_t rows,
                                             int64_t inner_size,
                                             bool x_broadcast_rows,
                                             bool y_broadcast_rows,
                                             musaStream_t stream) {
  if (rows == 0 || inner_size == 0) {
    return musaSuccess;
  }
  if (element_size == 4 && inner_size % 4 == 0 && IsAligned16(x) &&
      IsAligned16(y) && IsAligned16(output)) {
    const int64_t inner_vecs = inner_size / 4;
    const int x_blocks = BlocksForCount(inner_vecs);
    const int y_blocks =
        rows > 65535 ? 65535 : static_cast<int>(rows);
    WhereRowwiseFloat4Kernel<<<dim3(x_blocks, y_blocks, 1), kThreadsPerBlock,
                               0, stream>>>(
        condition, reinterpret_cast<const float4*>(x),
        reinterpret_cast<const float4*>(y), reinterpret_cast<float4*>(output),
        rows, inner_vecs, x_broadcast_rows, y_broadcast_rows);
    return musaGetLastError();
  }

  switch (element_size) {
    case 1:
      return LaunchWhereRowwiseTyped<uint8_t>(condition, x, y, output, rows,
                                              inner_size, x_broadcast_rows,
                                              y_broadcast_rows, stream);
    case 2:
      return LaunchWhereRowwiseTyped<uint16_t>(condition, x, y, output, rows,
                                               inner_size, x_broadcast_rows,
                                               y_broadcast_rows, stream);
    case 4:
      return LaunchWhereRowwiseTyped<uint32_t>(condition, x, y, output, rows,
                                               inner_size, x_broadcast_rows,
                                               y_broadcast_rows, stream);
    case 8:
      return LaunchWhereRowwiseTyped<uint64_t>(condition, x, y, output, rows,
                                               inner_size, x_broadcast_rows,
                                               y_broadcast_rows, stream);
    default:
      return musaErrorNotSupported;
  }
}
