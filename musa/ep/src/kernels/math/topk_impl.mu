#include "math/topk_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T>
__device__ __forceinline__ bool TopKValueGreater(T lhs, T rhs) {
  return MusaScalarToDouble(lhs) > MusaScalarToDouble(rhs);
}

template <typename T>
__device__ __forceinline__ bool TopKValueLess(T lhs, T rhs) {
  return MusaScalarToDouble(lhs) < MusaScalarToDouble(rhs);
}

template <typename T>
__device__ __forceinline__ bool TopKCandidateBetter(T candidate,
                                                    int64_t candidate_index,
                                                    T best, int64_t best_index,
                                                    bool has_best,
                                                    bool largest) {
  if (!has_best) {
    return true;
  }
  if (largest) {
    if (TopKValueGreater(candidate, best)) {
      return true;
    }
    if (TopKValueGreater(best, candidate)) {
      return false;
    }
  } else {
    if (TopKValueLess(candidate, best)) {
      return true;
    }
    if (TopKValueLess(best, candidate)) {
      return false;
    }
  }
  return candidate_index < best_index;
}

template <typename T>
__global__ void TopKKernel(const T* input, T* values, int64_t* indices,
                           MusaTopKParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.output_elements;
       output_index += total_threads) {
    const int64_t inner_index = output_index % params.inner;
    const int64_t k_index = (output_index / params.inner) % params.k;
    const int64_t outer_index = output_index / (params.k * params.inner);
    const int64_t input_base =
        outer_index * params.dim * params.inner + inner_index;

    T best{};
    int64_t best_index = 0;
    bool has_best = false;

    for (int64_t candidate_index = 0; candidate_index < params.dim;
         ++candidate_index) {
      const T candidate = input[input_base + candidate_index * params.inner];

      int64_t rank = 0;
      for (int64_t other_index = 0; other_index < params.dim; ++other_index) {
        if (other_index == candidate_index) {
          continue;
        }
        const T other = input[input_base + other_index * params.inner];
        bool other_before_candidate = false;
        if (params.largest) {
          if (TopKValueGreater(other, candidate)) {
            other_before_candidate = true;
          } else if (!TopKValueGreater(candidate, other) &&
                     other_index < candidate_index) {
            other_before_candidate = true;
          }
        } else {
          if (TopKValueLess(other, candidate)) {
            other_before_candidate = true;
          } else if (!TopKValueLess(candidate, other) &&
                     other_index < candidate_index) {
            other_before_candidate = true;
          }
        }
        if (other_before_candidate) {
          ++rank;
        }
      }

      if (rank == k_index &&
          TopKCandidateBetter(candidate, candidate_index, best, best_index,
                              has_best, params.largest != 0)) {
        best = candidate;
        best_index = candidate_index;
        has_best = true;
      }
    }

    values[output_index] = best;
    indices[output_index] = best_index;
  }
}

template <typename T>
musaError_t LaunchTopKTyped(const void* input, void* values, int64_t* indices,
                            MusaTopKParams params, musaStream_t stream) {
  if (params.output_elements == 0) {
    return musaSuccess;
  }
  TopKKernel<T>
      <<<BlocksForCount(params.output_elements), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const T*>(input), reinterpret_cast<T*>(values),
          indices, params);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaTopKKernel(const void* input, void* values,
                                 int64_t* indices, MusaTopKParams params,
                                 MusaElementType elem_type,
                                 musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchTopKTyped<float>(input, values, indices, params, stream);
    case MusaElementType::Double:
      return LaunchTopKTyped<double>(input, values, indices, params, stream);
    case MusaElementType::Float16:
      return LaunchTopKTyped<__half>(input, values, indices, params, stream);
    case MusaElementType::Int32:
      return LaunchTopKTyped<int32_t>(input, values, indices, params, stream);
    case MusaElementType::Int64:
      return LaunchTopKTyped<int64_t>(input, values, indices, params, stream);
    default:
      return musaErrorNotSupported;
  }
}
