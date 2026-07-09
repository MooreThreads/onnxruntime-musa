#include "tensor/unique_impl.h"

#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T>
__device__ __forceinline__ bool ValuesEqual(T lhs, T rhs) {
  return lhs == rhs;
}

template <typename T>
__device__ __forceinline__ bool ValueLess(T lhs, T rhs) {
  return lhs < rhs;
}

template <typename T>
__device__ __forceinline__ int IsFirstOccurrence(const T* input,
                                                 int64_t index) {
  const T value = input[index];
  for (int64_t i = 0; i < index; ++i) {
    if (ValuesEqual(input[i], value)) {
      return 0;
    }
  }
  return 1;
}

template <typename T>
__device__ int64_t UniquePosition(const T* input, const int* first_flags,
                                  int64_t input_count, int64_t index,
                                  int sorted) {
  const T value = input[index];
  int64_t pos = 0;
  for (int64_t i = 0; i < input_count; ++i) {
    if (first_flags[i] == 0) {
      continue;
    }
    if (sorted) {
      if (ValueLess(input[i], value)) {
        ++pos;
      }
    } else if (i < index) {
      ++pos;
    }
  }
  return pos;
}

template <typename T>
__device__ int64_t FirstOccurrenceIndex(const T* input, int64_t input_count,
                                        int64_t index) {
  const T value = input[index];
  for (int64_t i = 0; i < input_count; ++i) {
    if (ValuesEqual(input[i], value)) {
      return i;
    }
  }
  return index;
}

template <typename T>
__device__ int64_t OccurrenceCount(const T* input, int64_t input_count,
                                   T value) {
  int64_t count = 0;
  for (int64_t i = 0; i < input_count; ++i) {
    if (ValuesEqual(input[i], value)) {
      ++count;
    }
  }
  return count;
}

template <typename T>
__global__ void UniqueCountKernel(const T* input, int64_t count,
                                  int* first_flags, int* block_counts) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int first = index < count ? IsFirstOccurrence(input, index) : 0;
  if (index < count) {
    first_flags[index] = first;
  }

  __shared__ int shared[kThreadsPerBlock];
  shared[threadIdx.x] = first;
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
__global__ void UniqueOutputKernel(const T* input, int64_t input_count,
                                   int64_t unique_count,
                                   const int* first_flags, T* values,
                                   int64_t* indices,
                                   int64_t* inverse_indices, int64_t* counts,
                                   int sorted) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= input_count) {
    return;
  }

  const int64_t first_index = FirstOccurrenceIndex(input, input_count, index);
  const int64_t inverse_pos = UniquePosition(input, first_flags, input_count,
                                             first_index, sorted);
  if (inverse_indices != nullptr && inverse_pos < unique_count) {
    inverse_indices[index] = inverse_pos;
  }

  if (first_flags[index] == 0) {
    return;
  }

  const int64_t pos =
      UniquePosition(input, first_flags, input_count, index, sorted);
  if (pos >= unique_count) {
    return;
  }
  if (values != nullptr) {
    values[pos] = input[index];
  }
  if (indices != nullptr) {
    indices[pos] = index;
  }
  if (counts != nullptr) {
    counts[pos] = OccurrenceCount(input, input_count, input[index]);
  }
}

template <typename T>
musaError_t LaunchUniqueCountTyped(const void* input, int64_t count,
                                   int* first_flags, int* block_counts,
                                   musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  UniqueCountKernel<T><<<UniqueBlockCount(count), kThreadsPerBlock, 0,
                         stream>>>(reinterpret_cast<const T*>(input), count,
                                   first_flags, block_counts);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchUniqueOutputTyped(const void* input, int64_t input_count,
                                    int64_t unique_count,
                                    const int* first_flags, void* values,
                                    int64_t* indices,
                                    int64_t* inverse_indices, int64_t* counts,
                                    int sorted, musaStream_t stream) {
  if (input_count == 0 || unique_count == 0) {
    return musaSuccess;
  }
  UniqueOutputKernel<T><<<UniqueBlockCount(input_count), kThreadsPerBlock, 0,
                          stream>>>(
      reinterpret_cast<const T*>(input), input_count, unique_count, first_flags,
      reinterpret_cast<T*>(values), indices, inverse_indices, counts, sorted);
  return musaGetLastError();
}

}  // namespace

int UniqueBlockCount(int64_t total_elements) {
  return BlocksForCount(total_elements);
}

musaError_t LaunchMusaUniqueCountKernel(const void* input, int64_t count,
                                        int* first_flags, int* block_counts,
                                        MusaElementType elem_type,
                                        musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Int32:
      return LaunchUniqueCountTyped<int32_t>(input, count, first_flags,
                                             block_counts, stream);
    case MusaElementType::Int64:
      return LaunchUniqueCountTyped<int64_t>(input, count, first_flags,
                                             block_counts, stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaUniqueOutputKernel(
    const void* input, int64_t input_count, int64_t unique_count,
    const int* first_flags, void* values, int64_t* indices,
    int64_t* inverse_indices, int64_t* counts, int sorted,
    MusaElementType elem_type, musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Int32:
      return LaunchUniqueOutputTyped<int32_t>(
          input, input_count, unique_count, first_flags, values, indices,
          inverse_indices, counts, sorted, stream);
    case MusaElementType::Int64:
      return LaunchUniqueOutputTyped<int64_t>(
          input, input_count, unique_count, first_flags, values, indices,
          inverse_indices, counts, sorted, stream);
    default:
      return musaErrorNotSupported;
  }
}
