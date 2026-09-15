#include <cstdint>

#include "math/topk_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

constexpr int kTopKMaxBlockItems = static_cast<int>(kMusaTopKBlockSortMaxDim);
constexpr int kTopKPrefixThreads = kThreadsPerBlock;

template <typename T>
__device__ __forceinline__ bool TopKValueGreater(T lhs, T rhs) {
  return lhs > rhs;
}

template <>
__device__ __forceinline__ bool TopKValueGreater<__half>(__half lhs,
                                                         __half rhs) {
  return __half2float(lhs) > __half2float(rhs);
}

template <typename T>
__device__ __forceinline__ bool TopKValueLess(T lhs, T rhs) {
  return lhs < rhs;
}

template <>
__device__ __forceinline__ bool TopKValueLess<__half>(__half lhs, __half rhs) {
  return __half2float(lhs) < __half2float(rhs);
}

template <typename T>
__device__ __forceinline__ bool TopKValueEqual(T lhs, T rhs) {
  return lhs == rhs;
}

template <>
__device__ __forceinline__ bool TopKValueEqual<__half>(__half lhs, __half rhs) {
  return __half2float(lhs) == __half2float(rhs);
}

template <typename T>
__device__ __forceinline__ bool TopKPairBefore(T lhs, int64_t lhs_index, T rhs,
                                               int64_t rhs_index,
                                               bool largest) {
  if (largest) {
    if (TopKValueGreater(lhs, rhs)) {
      return true;
    }
    if (TopKValueGreater(rhs, lhs)) {
      return false;
    }
  } else {
    if (TopKValueLess(lhs, rhs)) {
      return true;
    }
    if (TopKValueLess(rhs, lhs)) {
      return false;
    }
  }
  return lhs_index < rhs_index;
}

template <typename T>
__device__ __forceinline__ bool TopKEntryBefore(T lhs, int64_t lhs_index,
                                                bool lhs_valid, T rhs,
                                                int64_t rhs_index,
                                                bool rhs_valid, bool largest) {
  if (lhs_valid != rhs_valid) {
    return lhs_valid;
  }
  if (!lhs_valid) {
    return false;
  }
  return TopKPairBefore(lhs, lhs_index, rhs, rhs_index, largest);
}

__host__ __forceinline__ int TopKNextPowerOfTwo(int64_t value) {
  int result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

template <typename T>
__global__ void TopKPairReduceKernel(const T* input, T* values,
                                     int64_t* indices, MusaTopKParams params) {
  __shared__ T shared_values[kThreadsPerBlock];
  __shared__ int64_t shared_indices[kThreadsPerBlock];
  __shared__ bool shared_valid[kThreadsPerBlock];

  for (int64_t row = static_cast<int64_t>(blockIdx.x); row < params.rows;
       row += gridDim.x) {
    const int64_t inner_index = row % params.inner;
    const int64_t outer_index = row / params.inner;
    const int64_t input_base =
        outer_index * params.dim * params.inner + inner_index;

    T best{};
    int64_t best_index = 0;
    bool has_best = false;
    for (int64_t candidate = threadIdx.x; candidate < params.dim;
         candidate += blockDim.x) {
      const T value = input[input_base + candidate * params.inner];
      if (!has_best || TopKPairBefore(value, candidate, best, best_index,
                                      params.largest != 0)) {
        best = value;
        best_index = candidate;
        has_best = true;
      }
    }

    shared_values[threadIdx.x] = best;
    shared_indices[threadIdx.x] = best_index;
    shared_valid[threadIdx.x] = has_best;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        const int other = static_cast<int>(threadIdx.x) + stride;
        if (shared_valid[other] &&
            (!shared_valid[threadIdx.x] ||
             TopKPairBefore(shared_values[other], shared_indices[other],
                            shared_values[threadIdx.x],
                            shared_indices[threadIdx.x],
                            params.largest != 0))) {
          shared_values[threadIdx.x] = shared_values[other];
          shared_indices[threadIdx.x] = shared_indices[other];
          shared_valid[threadIdx.x] = true;
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      const int64_t output_index = outer_index * params.inner + inner_index;
      values[output_index] = shared_values[0];
      indices[output_index] = shared_indices[0];
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void TopKBlockSortKernel(const T* input, T* values, int64_t* indices,
                                    MusaTopKParams params, int sort_size) {
  __shared__ T shared_values[kTopKMaxBlockItems];
  __shared__ int64_t shared_indices[kTopKMaxBlockItems];
  __shared__ bool shared_valid[kTopKMaxBlockItems];

  for (int64_t row = static_cast<int64_t>(blockIdx.x); row < params.rows;
       row += gridDim.x) {
    const int64_t inner_index = row % params.inner;
    const int64_t outer_index = row / params.inner;
    const int64_t input_base =
        outer_index * params.dim * params.inner + inner_index;

    for (int item = static_cast<int>(threadIdx.x); item < sort_size;
         item += blockDim.x) {
      if (item < params.dim) {
        shared_values[item] = input[input_base + item * params.inner];
        shared_indices[item] = item;
        shared_valid[item] = true;
      } else {
        shared_values[item] = T{};
        shared_indices[item] = item;
        shared_valid[item] = false;
      }
    }
    __syncthreads();

    for (int size = 2; size <= sort_size; size <<= 1) {
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        for (int item = static_cast<int>(threadIdx.x); item < sort_size;
             item += blockDim.x) {
          const int other = item ^ stride;
          if (other > item) {
            const bool low_half = (item & size) == 0;
            const bool lhs_before = TopKEntryBefore(
                shared_values[item], shared_indices[item], shared_valid[item],
                shared_values[other], shared_indices[other],
                shared_valid[other], params.largest != 0);
            const bool swap = low_half ? !lhs_before : lhs_before;
            if (swap) {
              const T value = shared_values[item];
              const int64_t index = shared_indices[item];
              const bool valid = shared_valid[item];
              shared_values[item] = shared_values[other];
              shared_indices[item] = shared_indices[other];
              shared_valid[item] = shared_valid[other];
              shared_values[other] = value;
              shared_indices[other] = index;
              shared_valid[other] = valid;
            }
          }
        }
        __syncthreads();
      }
    }

    for (int64_t k_index = threadIdx.x; k_index < params.k;
         k_index += blockDim.x) {
      const int64_t output_index = outer_index * params.k * params.inner +
                                   k_index * params.inner + inner_index;
      values[output_index] = shared_values[k_index];
      indices[output_index] = shared_indices[k_index];
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void TopKStablePostprocessKernel(const T* input, T* values,
                                            int64_t* indices,
                                            MusaTopKParams params,
                                            int sort_size) {
  __shared__ T shared_values[kTopKMaxBlockItems];
  __shared__ int64_t shared_indices[kTopKMaxBlockItems];
  __shared__ bool shared_valid[kTopKMaxBlockItems];
  __shared__ int prefix[kTopKPrefixThreads];
  __shared__ int threshold_begin;
  __shared__ int selected_threshold;
  __shared__ T threshold;

  for (int64_t row = static_cast<int64_t>(blockIdx.x); row < params.rows;
       row += gridDim.x) {
    const int64_t inner_index = row % params.inner;
    const int64_t outer_index = row / params.inner;
    const int64_t input_base =
        outer_index * params.dim * params.inner + inner_index;
    const int64_t output_base =
        outer_index * params.k * params.inner + inner_index;

    for (int item = static_cast<int>(threadIdx.x); item < sort_size;
         item += blockDim.x) {
      if (item < params.k) {
        const int64_t output_index = output_base + item * params.inner;
        shared_values[item] = values[output_index];
        shared_indices[item] = indices[output_index];
        shared_valid[item] = true;
      } else {
        shared_values[item] = T{};
        shared_indices[item] = item;
        shared_valid[item] = false;
      }
    }

    if (threadIdx.x == 0) {
      threshold = values[output_base + (params.k - 1) * params.inner];
      threshold_begin = static_cast<int>(params.k - 1);
      while (threshold_begin > 0 &&
             TopKValueEqual(
                 values[output_base + (threshold_begin - 1) * params.inner],
                 threshold)) {
        --threshold_begin;
      }
      selected_threshold = 0;
    }
    __syncthreads();

    const int threshold_quota = static_cast<int>(params.k) - threshold_begin;
    for (int64_t tile = 0; tile < params.dim; tile += blockDim.x) {
      const int64_t candidate = tile + threadIdx.x;
      const bool matches =
          candidate < params.dim &&
          TopKValueEqual(input[input_base + candidate * params.inner],
                         threshold);
      prefix[threadIdx.x] = matches ? 1 : 0;
      __syncthreads();

      for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        const int add =
            threadIdx.x >= offset ? prefix[threadIdx.x - offset] : 0;
        __syncthreads();
        prefix[threadIdx.x] += add;
        __syncthreads();
      }

      const int selected_before = selected_threshold;
      if (matches) {
        const int rank = selected_before + prefix[threadIdx.x] - 1;
        if (rank < threshold_quota) {
          shared_indices[threshold_begin + rank] = candidate;
        }
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        selected_threshold += prefix[blockDim.x - 1];
      }
      __syncthreads();
      if (selected_threshold >= threshold_quota) {
        break;
      }
    }

    for (int size = 2; size <= sort_size; size <<= 1) {
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        for (int item = static_cast<int>(threadIdx.x); item < sort_size;
             item += blockDim.x) {
          const int other = item ^ stride;
          if (other > item) {
            const bool low_half = (item & size) == 0;
            const bool lhs_before = TopKEntryBefore(
                shared_values[item], shared_indices[item], shared_valid[item],
                shared_values[other], shared_indices[other],
                shared_valid[other], params.largest != 0);
            const bool swap = low_half ? !lhs_before : lhs_before;
            if (swap) {
              const T value = shared_values[item];
              const int64_t index = shared_indices[item];
              const bool valid = shared_valid[item];
              shared_values[item] = shared_values[other];
              shared_indices[item] = shared_indices[other];
              shared_valid[item] = shared_valid[other];
              shared_values[other] = value;
              shared_indices[other] = index;
              shared_valid[other] = valid;
            }
          }
        }
        __syncthreads();
      }
    }

    for (int64_t k_index = threadIdx.x; k_index < params.k;
         k_index += blockDim.x) {
      const int64_t output_index = output_base + k_index * params.inner;
      values[output_index] = shared_values[k_index];
      indices[output_index] = shared_indices[k_index];
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void TopKGenericKernel(const T* input, T* values, int64_t* indices,
                                  MusaTopKParams params) {
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       row < params.rows; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t inner_index = row % params.inner;
    const int64_t outer_index = row / params.inner;
    const int64_t input_base =
        outer_index * params.dim * params.inner + inner_index;
    const int64_t output_base =
        outer_index * params.k * params.inner + inner_index;

    for (int64_t k_index = 0; k_index < params.k; ++k_index) {
      indices[output_base + k_index * params.inner] = -1;
    }
    for (int64_t candidate = 0; candidate < params.dim; ++candidate) {
      const T candidate_value = input[input_base + candidate * params.inner];
      int64_t insert = params.k;
      for (int64_t k_index = 0; k_index < params.k; ++k_index) {
        const int64_t output_index = output_base + k_index * params.inner;
        if (indices[output_index] < 0 ||
            TopKPairBefore(candidate_value, candidate, values[output_index],
                           indices[output_index], params.largest != 0)) {
          insert = k_index;
          break;
        }
      }
      if (insert < params.k) {
        for (int64_t k_index = params.k - 1; k_index > insert; --k_index) {
          const int64_t dst = output_base + k_index * params.inner;
          const int64_t src = output_base + (k_index - 1) * params.inner;
          values[dst] = values[src];
          indices[dst] = indices[src];
        }
        values[output_base + insert * params.inner] = candidate_value;
        indices[output_base + insert * params.inner] = candidate;
      }
    }
  }
}

template <typename T>
musaError_t LaunchPairReduceTyped(const void* input, void* values,
                                  int64_t* indices, MusaTopKParams params,
                                  musaStream_t stream) {
  if (params.output_elements == 0) {
    return musaSuccess;
  }
  const int blocks =
      static_cast<int>(params.rows > kMaxBlocks ? kMaxBlocks : params.rows);
  TopKPairReduceKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input), reinterpret_cast<T*>(values), indices,
      params);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchGenericTopKTyped(const void* input, void* values,
                                   int64_t* indices, MusaTopKParams params,
                                   musaStream_t stream) {
  if (params.output_elements == 0) {
    return musaSuccess;
  }
  const int blocks = static_cast<int>(
      (params.rows + kThreadsPerBlock - 1) / kThreadsPerBlock > kMaxBlocks
          ? kMaxBlocks
          : (params.rows + kThreadsPerBlock - 1) / kThreadsPerBlock);
  TopKGenericKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input), reinterpret_cast<T*>(values), indices,
      params);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchBlockSortTyped(const void* input, void* values,
                                 int64_t* indices, MusaTopKParams params,
                                 musaStream_t stream) {
  if (params.output_elements == 0) {
    return musaSuccess;
  }
  if (params.dim > kMusaTopKBlockSortMaxDim) {
    return musaErrorNotSupported;
  }
  const int sort_size = TopKNextPowerOfTwo(params.dim);
  const int threads =
      sort_size < 32
          ? 32
          : (sort_size < kThreadsPerBlock ? sort_size : kThreadsPerBlock);
  const int blocks =
      static_cast<int>(params.rows > kMaxBlocks ? kMaxBlocks : params.rows);
  TopKBlockSortKernel<T><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const T*>(input), reinterpret_cast<T*>(values), indices,
      params, sort_size);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchStablePostprocessTyped(const void* input, void* values,
                                         int64_t* indices,
                                         MusaTopKParams params,
                                         musaStream_t stream) {
  if (params.output_elements == 0) {
    return musaSuccess;
  }
  if (params.k > kMusaTopKStablePostprocessMaxK) {
    return musaErrorNotSupported;
  }
  const int sort_size = TopKNextPowerOfTwo(params.k);
  const int blocks =
      static_cast<int>(params.rows > kMaxBlocks ? kMaxBlocks : params.rows);
  TopKStablePostprocessKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input), reinterpret_cast<T*>(values), indices,
      params, sort_size);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaTopKPairReduceKernel(const void* input, void* values,
                                           int64_t* indices,
                                           MusaTopKParams params,
                                           MusaElementType elem_type,
                                           musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Uint8:
      return LaunchPairReduceTyped<uint8_t>(input, values, indices, params,
                                            stream);
    case MusaElementType::Uint16:
      return LaunchPairReduceTyped<uint16_t>(input, values, indices, params,
                                             stream);
    case MusaElementType::Uint32:
      return LaunchPairReduceTyped<uint32_t>(input, values, indices, params,
                                             stream);
    case MusaElementType::Uint64:
      return LaunchPairReduceTyped<uint64_t>(input, values, indices, params,
                                             stream);
    case MusaElementType::Int8:
      return LaunchPairReduceTyped<int8_t>(input, values, indices, params,
                                           stream);
    case MusaElementType::Int16:
      return LaunchPairReduceTyped<int16_t>(input, values, indices, params,
                                            stream);
    case MusaElementType::Float:
      return LaunchPairReduceTyped<float>(input, values, indices, params,
                                          stream);
    case MusaElementType::Double:
      return LaunchPairReduceTyped<double>(input, values, indices, params,
                                           stream);
    case MusaElementType::Float16:
      return LaunchPairReduceTyped<__half>(input, values, indices, params,
                                           stream);
    case MusaElementType::Int32:
      return LaunchPairReduceTyped<int32_t>(input, values, indices, params,
                                            stream);
    case MusaElementType::Int64:
      return LaunchPairReduceTyped<int64_t>(input, values, indices, params,
                                            stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaTopKBlockSortKernel(const void* input, void* values,
                                          int64_t* indices,
                                          MusaTopKParams params,
                                          MusaElementType elem_type,
                                          musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Uint8:
      return LaunchBlockSortTyped<uint8_t>(input, values, indices, params,
                                           stream);
    case MusaElementType::Uint16:
      return LaunchBlockSortTyped<uint16_t>(input, values, indices, params,
                                            stream);
    case MusaElementType::Uint32:
      return LaunchBlockSortTyped<uint32_t>(input, values, indices, params,
                                            stream);
    case MusaElementType::Uint64:
      return LaunchBlockSortTyped<uint64_t>(input, values, indices, params,
                                            stream);
    case MusaElementType::Int8:
      return LaunchBlockSortTyped<int8_t>(input, values, indices, params,
                                          stream);
    case MusaElementType::Int16:
      return LaunchBlockSortTyped<int16_t>(input, values, indices, params,
                                           stream);
    case MusaElementType::Float:
      return LaunchBlockSortTyped<float>(input, values, indices, params,
                                         stream);
    case MusaElementType::Double:
      return LaunchBlockSortTyped<double>(input, values, indices, params,
                                          stream);
    case MusaElementType::Float16:
      return LaunchBlockSortTyped<__half>(input, values, indices, params,
                                          stream);
    case MusaElementType::Int32:
      return LaunchBlockSortTyped<int32_t>(input, values, indices, params,
                                           stream);
    case MusaElementType::Int64:
      return LaunchBlockSortTyped<int64_t>(input, values, indices, params,
                                           stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaTopKStablePostprocessKernel(
    const void* input, void* values, int64_t* indices, MusaTopKParams params,
    MusaElementType elem_type, musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Uint8:
      return LaunchStablePostprocessTyped<uint8_t>(input, values, indices,
                                                   params, stream);
    case MusaElementType::Uint16:
      return LaunchStablePostprocessTyped<uint16_t>(input, values, indices,
                                                    params, stream);
    case MusaElementType::Uint32:
      return LaunchStablePostprocessTyped<uint32_t>(input, values, indices,
                                                    params, stream);
    case MusaElementType::Uint64:
      return LaunchStablePostprocessTyped<uint64_t>(input, values, indices,
                                                    params, stream);
    case MusaElementType::Int8:
      return LaunchStablePostprocessTyped<int8_t>(input, values, indices,
                                                  params, stream);
    case MusaElementType::Int16:
      return LaunchStablePostprocessTyped<int16_t>(input, values, indices,
                                                   params, stream);
    case MusaElementType::Float:
      return LaunchStablePostprocessTyped<float>(input, values, indices, params,
                                                 stream);
    case MusaElementType::Double:
      return LaunchStablePostprocessTyped<double>(input, values, indices,
                                                  params, stream);
    case MusaElementType::Float16:
      return LaunchStablePostprocessTyped<__half>(input, values, indices,
                                                  params, stream);
    case MusaElementType::Int32:
      return LaunchStablePostprocessTyped<int32_t>(input, values, indices,
                                                   params, stream);
    case MusaElementType::Int64:
      return LaunchStablePostprocessTyped<int64_t>(input, values, indices,
                                                   params, stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaTopKGenericKernel(const void* input, void* values,
                                        int64_t* indices,
                                        MusaTopKParams params,
                                        MusaElementType elem_type,
                                        musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Uint8:
      return LaunchGenericTopKTyped<uint8_t>(input, values, indices, params,
                                             stream);
    case MusaElementType::Uint16:
      return LaunchGenericTopKTyped<uint16_t>(input, values, indices, params,
                                              stream);
    case MusaElementType::Uint32:
      return LaunchGenericTopKTyped<uint32_t>(input, values, indices, params,
                                              stream);
    case MusaElementType::Uint64:
      return LaunchGenericTopKTyped<uint64_t>(input, values, indices, params,
                                              stream);
    case MusaElementType::Int8:
      return LaunchGenericTopKTyped<int8_t>(input, values, indices, params,
                                            stream);
    case MusaElementType::Int16:
      return LaunchGenericTopKTyped<int16_t>(input, values, indices, params,
                                             stream);
    case MusaElementType::Float:
      return LaunchGenericTopKTyped<float>(input, values, indices, params,
                                           stream);
    case MusaElementType::Double:
      return LaunchGenericTopKTyped<double>(input, values, indices, params,
                                            stream);
    case MusaElementType::Float16:
      return LaunchGenericTopKTyped<__half>(input, values, indices, params,
                                            stream);
    case MusaElementType::Int32:
      return LaunchGenericTopKTyped<int32_t>(input, values, indices, params,
                                             stream);
    case MusaElementType::Int64:
      return LaunchGenericTopKTyped<int64_t>(input, values, indices, params,
                                             stream);
    default:
      return musaErrorNotSupported;
  }
}
