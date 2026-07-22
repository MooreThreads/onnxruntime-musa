#include <cstdint>
#include <cub/cub.cuh>
#include <limits>

#include "math/topk_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

constexpr int kTopKMaxBlockItems = static_cast<int>(kMusaTopKBlockSortMaxDim);
constexpr int kTopKPrefixThreads = kThreadsPerBlock;
constexpr size_t kTopKWorkspaceAlignment = 256;

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
__global__ void TopKFillRadixInputKernel(const T* input, T* keys,
                                         int64_t* source_indices,
                                         MusaTopKParams params) {
  const int64_t total = params.rows * params.dim;
  for (int64_t linear =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = linear / params.dim;
    const int64_t dim_index = linear - row * params.dim;
    const int64_t inner_index = row % params.inner;
    const int64_t outer_index = row / params.inner;
    const int64_t input_index = outer_index * params.dim * params.inner +
                                dim_index * params.inner + inner_index;
    keys[linear] = input[input_index];
    source_indices[linear] = dim_index;
  }
}

__global__ void TopKFillSegmentOffsetsKernel(int* offsets, int rows, int dim) {
  for (int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       index <= rows; index += blockDim.x * gridDim.x) {
    offsets[index] = index * dim;
  }
}

template <typename T>
__global__ void TopKCopyRadixOutputKernel(const T* sorted_values,
                                          const int64_t* sorted_indices,
                                          T* values, int64_t* indices,
                                          MusaTopKParams params) {
  const int64_t total = params.rows * params.k;
  for (int64_t linear =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = linear / params.k;
    const int64_t k_index = linear - row * params.k;
    const int64_t inner_index = row % params.inner;
    const int64_t outer_index = row / params.inner;
    const int64_t output_index = outer_index * params.k * params.inner +
                                 k_index * params.inner + inner_index;
    const int64_t sorted_index = row * params.dim + k_index;
    values[output_index] = sorted_values[sorted_index];
    indices[output_index] = sorted_indices[sorted_index];
  }
}

__host__ __forceinline__ size_t TopKAlignWorkspace(size_t offset) {
  return (offset + kTopKWorkspaceAlignment - 1) &
         ~(kTopKWorkspaceAlignment - 1);
}

template <typename T>
musaError_t GetRadixTempBytes(MusaTopKParams params, size_t* temp_bytes) {
  if (params.rows > std::numeric_limits<int>::max() ||
      params.dim > std::numeric_limits<int>::max() ||
      params.rows > std::numeric_limits<int>::max() / params.dim) {
    return musaErrorNotSupported;
  }
  const int total = static_cast<int>(params.rows * params.dim);
  const int rows = static_cast<int>(params.rows);
  return cub::DeviceSegmentedRadixSort::SortPairs(
      nullptr, *temp_bytes, static_cast<const T*>(nullptr),
      static_cast<T*>(nullptr), static_cast<const int64_t*>(nullptr),
      static_cast<int64_t*>(nullptr), total, rows,
      static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), 0,
      static_cast<int>(sizeof(T) * 8));
}

template <typename T>
musaError_t GetRadixWorkspaceSizeTyped(MusaTopKParams params,
                                       size_t* workspace_bytes) {
  size_t temp_bytes = 0;
  const musaError_t status = GetRadixTempBytes<T>(params, &temp_bytes);
  if (status != musaSuccess) {
    return status;
  }
  const size_t total = static_cast<size_t>(params.rows * params.dim);
  size_t offset = 0;
  offset = TopKAlignWorkspace(offset) + total * sizeof(T);
  offset = TopKAlignWorkspace(offset) + total * sizeof(T);
  offset = TopKAlignWorkspace(offset) + total * sizeof(int64_t);
  offset = TopKAlignWorkspace(offset) + total * sizeof(int64_t);
  offset = TopKAlignWorkspace(offset) +
           static_cast<size_t>(params.rows + 1) * sizeof(int);
  offset = TopKAlignWorkspace(offset) + temp_bytes;
  *workspace_bytes = offset;
  return musaSuccess;
}

template <typename T>
musaError_t LaunchRadixSortTyped(const void* input, void* values,
                                 int64_t* indices, MusaTopKParams params,
                                 void* workspace, size_t workspace_bytes,
                                 musaStream_t stream) {
  size_t required_bytes = 0;
  musaError_t status = GetRadixWorkspaceSizeTyped<T>(params, &required_bytes);
  if (status != musaSuccess) {
    return status;
  }
  if (workspace == nullptr || workspace_bytes < required_bytes) {
    return musaErrorInvalidValue;
  }

  size_t temp_bytes = 0;
  status = GetRadixTempBytes<T>(params, &temp_bytes);
  if (status != musaSuccess) {
    return status;
  }

  const size_t total = static_cast<size_t>(params.rows * params.dim);
  auto* base = reinterpret_cast<unsigned char*>(workspace);
  size_t offset = 0;
  offset = TopKAlignWorkspace(offset);
  T* keys_in = reinterpret_cast<T*>(base + offset);
  offset += total * sizeof(T);
  offset = TopKAlignWorkspace(offset);
  T* keys_out = reinterpret_cast<T*>(base + offset);
  offset += total * sizeof(T);
  offset = TopKAlignWorkspace(offset);
  int64_t* indices_in = reinterpret_cast<int64_t*>(base + offset);
  offset += total * sizeof(int64_t);
  offset = TopKAlignWorkspace(offset);
  int64_t* indices_out = reinterpret_cast<int64_t*>(base + offset);
  offset += total * sizeof(int64_t);
  offset = TopKAlignWorkspace(offset);
  int* segment_offsets = reinterpret_cast<int*>(base + offset);
  offset += static_cast<size_t>(params.rows + 1) * sizeof(int);
  offset = TopKAlignWorkspace(offset);
  void* temp_storage = base + offset;

  const int total_blocks = static_cast<int>(
      (total + kThreadsPerBlock - 1) / kThreadsPerBlock > kMaxBlocks
          ? kMaxBlocks
          : (total + kThreadsPerBlock - 1) / kThreadsPerBlock);
  TopKFillRadixInputKernel<T><<<total_blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input), keys_in, indices_in, params);
  const int offset_blocks = static_cast<int>(
      (params.rows + 1 + kThreadsPerBlock - 1) / kThreadsPerBlock > kMaxBlocks
          ? kMaxBlocks
          : (params.rows + 1 + kThreadsPerBlock - 1) / kThreadsPerBlock);
  TopKFillSegmentOffsetsKernel<<<offset_blocks, kThreadsPerBlock, 0, stream>>>(
      segment_offsets, static_cast<int>(params.rows),
      static_cast<int>(params.dim));
  status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }

  const int total_items = static_cast<int>(params.rows * params.dim);
  const int rows = static_cast<int>(params.rows);
  if (params.largest != 0) {
    status = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage, temp_bytes, keys_in, keys_out, indices_in, indices_out,
        total_items, rows, segment_offsets, segment_offsets + 1, 0,
        static_cast<int>(sizeof(T) * 8), stream);
  } else {
    status = cub::DeviceSegmentedRadixSort::SortPairs(
        temp_storage, temp_bytes, keys_in, keys_out, indices_in, indices_out,
        total_items, rows, segment_offsets, segment_offsets + 1, 0,
        static_cast<int>(sizeof(T) * 8), stream);
  }
  if (status != musaSuccess) {
    return status;
  }

  const int64_t output_total = params.rows * params.k;
  const int output_blocks = static_cast<int>(
      (output_total + kThreadsPerBlock - 1) / kThreadsPerBlock > kMaxBlocks
          ? kMaxBlocks
          : (output_total + kThreadsPerBlock - 1) / kThreadsPerBlock);
  TopKCopyRadixOutputKernel<T><<<output_blocks, kThreadsPerBlock, 0, stream>>>(
      keys_out, indices_out, reinterpret_cast<T*>(values), indices, params);
  return musaGetLastError();
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

musaError_t GetMusaTopKRadixSortWorkspaceSize(MusaTopKParams params,
                                              MusaElementType elem_type,
                                              size_t* workspace_bytes) {
  if (workspace_bytes == nullptr) {
    return musaErrorInvalidValue;
  }
  switch (elem_type) {
    case MusaElementType::Float:
      return GetRadixWorkspaceSizeTyped<float>(params, workspace_bytes);
    case MusaElementType::Double:
      return GetRadixWorkspaceSizeTyped<double>(params, workspace_bytes);
    case MusaElementType::Float16:
      return GetRadixWorkspaceSizeTyped<__half>(params, workspace_bytes);
    case MusaElementType::Int32:
      return GetRadixWorkspaceSizeTyped<int32_t>(params, workspace_bytes);
    case MusaElementType::Int64:
      return GetRadixWorkspaceSizeTyped<int64_t>(params, workspace_bytes);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaTopKRadixSortKernel(
    const void* input, void* values, int64_t* indices, MusaTopKParams params,
    MusaElementType elem_type, void* workspace, size_t workspace_bytes,
    musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchRadixSortTyped<float>(input, values, indices, params,
                                         workspace, workspace_bytes, stream);
    case MusaElementType::Double:
      return LaunchRadixSortTyped<double>(input, values, indices, params,
                                          workspace, workspace_bytes, stream);
    case MusaElementType::Float16:
      return LaunchRadixSortTyped<__half>(input, values, indices, params,
                                          workspace, workspace_bytes, stream);
    case MusaElementType::Int32:
      return LaunchRadixSortTyped<int32_t>(input, values, indices, params,
                                           workspace, workspace_bytes, stream);
    case MusaElementType::Int64:
      return LaunchRadixSortTyped<int64_t>(input, values, indices, params,
                                           workspace, workspace_bytes, stream);
    default:
      return musaErrorNotSupported;
  }
}
