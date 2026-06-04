#include "tensor/nonzero_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T>
__device__ __forceinline__ bool IsNonZeroValue(T value) {
  return value != static_cast<T>(0);
}

template <>
__device__ __forceinline__ bool IsNonZeroValue<__half>(__half value) {
  return __half2float(value) != 0.0f;
}

template <typename T>
__global__ void NonZeroCountKernel(const T* input,
                                   int64_t total_elements,
                                   int* block_counts) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int nz =
      index < total_elements && IsNonZeroValue(input[index]) ? 1 : 0;
  __shared__ int shared[kThreadsPerBlock];
  shared[threadIdx.x] = nz;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    block_counts[blockIdx.x] = shared[0];
  }
}

template <typename T>
__global__ void NonZeroOutputKernel(const T* input,
                                    const int* prefix_counts,
                                    int64_t* output,
                                    MusaNonZeroParams params) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int nz =
      index < params.total_elements && IsNonZeroValue(input[index]) ? 1 : 0;
  __shared__ int scan[kThreadsPerBlock];
  scan[threadIdx.x] = nz;
  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    int value = 0;
    if (threadIdx.x >= offset) {
      value = scan[threadIdx.x - offset];
    }
    __syncthreads();
    scan[threadIdx.x] += value;
    __syncthreads();
  }

  if (nz == 0) {
    return;
  }

  const int64_t block_base =
      blockIdx.x == 0 ? 0 : static_cast<int64_t>(prefix_counts[blockIdx.x - 1]);
  const int64_t output_pos = block_base + scan[threadIdx.x] - 1;
  int64_t remaining = index;
  for (int32_t axis = 0; axis < params.rank; ++axis) {
    const int64_t stride = params.input_strides[axis];
    const int64_t coord = stride == 0 ? 0 : remaining / stride;
    remaining = stride == 0 ? 0 : remaining - coord * stride;
    output[static_cast<int64_t>(axis) * params.nonzero_elements + output_pos] =
        coord;
  }
}

template <typename T>
musaError_t LaunchNonZeroCountTyped(const void* input,
                                    int64_t total_elements,
                                    int* block_counts,
                                    musaStream_t stream) {
  if (total_elements == 0) {
    return musaSuccess;
  }
  NonZeroCountKernel<T>
      <<<NonZeroBlockCount(total_elements), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const T*>(input), total_elements, block_counts);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchNonZeroOutputTyped(const void* input,
                                     const int* prefix_counts,
                                     int64_t* output,
                                     MusaNonZeroParams params,
                                     musaStream_t stream) {
  if (params.total_elements == 0 || params.nonzero_elements == 0) {
    return musaSuccess;
  }
  NonZeroOutputKernel<T>
      <<<NonZeroBlockCount(params.total_elements), kThreadsPerBlock, 0,
         stream>>>(reinterpret_cast<const T*>(input), prefix_counts, output,
                   params);
  return musaGetLastError();
}

}  // namespace

int NonZeroBlockCount(int64_t total_elements) {
  return BlocksForCount(total_elements);
}

musaError_t LaunchMusaNonZeroCountKernel(const void* input,
                                         int64_t total_elements,
                                         int* block_counts,
                                         MusaElementType elem_type,
                                         musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Bool:
    case MusaElementType::Uint8:
      return LaunchNonZeroCountTyped<uint8_t>(input, total_elements,
                                              block_counts, stream);
    case MusaElementType::Int32:
      return LaunchNonZeroCountTyped<int32_t>(input, total_elements,
                                              block_counts, stream);
    case MusaElementType::Int64:
      return LaunchNonZeroCountTyped<int64_t>(input, total_elements,
                                              block_counts, stream);
    case MusaElementType::Float:
      return LaunchNonZeroCountTyped<float>(input, total_elements,
                                            block_counts, stream);
    case MusaElementType::Float16:
      return LaunchNonZeroCountTyped<__half>(input, total_elements,
                                             block_counts, stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaNonZeroOutputKernel(const void* input,
                                          const int* prefix_counts,
                                          int64_t* output,
                                          MusaNonZeroParams params,
                                          MusaElementType elem_type,
                                          musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Bool:
    case MusaElementType::Uint8:
      return LaunchNonZeroOutputTyped<uint8_t>(input, prefix_counts, output,
                                               params, stream);
    case MusaElementType::Int32:
      return LaunchNonZeroOutputTyped<int32_t>(input, prefix_counts, output,
                                               params, stream);
    case MusaElementType::Int64:
      return LaunchNonZeroOutputTyped<int64_t>(input, prefix_counts, output,
                                               params, stream);
    case MusaElementType::Float:
      return LaunchNonZeroOutputTyped<float>(input, prefix_counts, output,
                                             params, stream);
    case MusaElementType::Float16:
      return LaunchNonZeroOutputTyped<__half>(input, prefix_counts, output,
                                              params, stream);
    default:
      return musaErrorNotSupported;
  }
}
