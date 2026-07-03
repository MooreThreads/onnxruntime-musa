#include "math/topk_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

// Treat tiny float32 backend differences as ties so ONNX lower-index
// tie-breaking remains stable around TopK selection thresholds.
__device__ __forceinline__ double TopKFloatTolerance(double lhs, double rhs) {
  return 1.0e-7 * fmax(1.0, fmax(fabs(lhs), fabs(rhs)));
}

template <typename T>
__device__ __forceinline__ double TopKComparisonTolerance(T, T) {
  return 0.0;
}

template <>
__device__ __forceinline__ double TopKComparisonTolerance<float>(float lhs,
                                                                 float rhs) {
  return TopKFloatTolerance(static_cast<double>(lhs),
                            static_cast<double>(rhs));
}

template <typename T>
__device__ __forceinline__ bool TopKValueGreater(T lhs, T rhs) {
  const double lhs_value = MusaScalarToDouble(lhs);
  const double rhs_value = MusaScalarToDouble(rhs);
  return lhs_value > rhs_value + TopKComparisonTolerance(lhs, rhs);
}

template <typename T>
__device__ __forceinline__ bool TopKValueLess(T lhs, T rhs) {
  const double lhs_value = MusaScalarToDouble(lhs);
  const double rhs_value = MusaScalarToDouble(rhs);
  return lhs_value + TopKComparisonTolerance(lhs, rhs) < rhs_value;
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
__device__ __forceinline__ bool TopKCandidateAfterPrevious(
    T candidate, int64_t candidate_index, T previous, int64_t previous_index,
    bool has_previous, bool largest) {
  if (!has_previous) {
    return true;
  }
  return TopKPairBefore(previous, previous_index, candidate, candidate_index,
                        largest);
}

template <typename T>
__global__ void TopKStableKernel(const T* input, T* values, int64_t* indices,
                                 MusaTopKParams params) {
  __shared__ T shared_values[kThreadsPerBlock];
  __shared__ int64_t shared_indices[kThreadsPerBlock];
  __shared__ bool shared_has[kThreadsPerBlock];
  __shared__ T previous;
  __shared__ int64_t previous_index;
  __shared__ bool has_previous;

  for (int64_t slice_index = static_cast<int64_t>(blockIdx.x);
       slice_index < params.rows; slice_index += gridDim.x) {
    const int64_t inner_index = slice_index % params.inner;
    const int64_t outer_index = slice_index / params.inner;
    const int64_t input_base =
        outer_index * params.dim * params.inner + inner_index;

    if (threadIdx.x == 0) {
      previous = T{};
      previous_index = 0;
      has_previous = false;
    }
    __syncthreads();

    for (int64_t k_index = 0; k_index < params.k; ++k_index) {
      T best{};
      int64_t best_index = 0;
      bool has_best = false;

      for (int64_t candidate_index = threadIdx.x; candidate_index < params.dim;
           candidate_index += blockDim.x) {
        const T candidate = input[input_base + candidate_index * params.inner];
        if (!TopKCandidateAfterPrevious(candidate, candidate_index, previous,
                                        previous_index, has_previous,
                                        params.largest != 0)) {
          continue;
        }
        if (!has_best ||
            TopKPairBefore(candidate, candidate_index, best, best_index,
                           params.largest != 0)) {
          best = candidate;
          best_index = candidate_index;
          has_best = true;
        }
      }

      shared_values[threadIdx.x] = best;
      shared_indices[threadIdx.x] = best_index;
      shared_has[threadIdx.x] = has_best;
      __syncthreads();

      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          const int other = threadIdx.x + stride;
          if (shared_has[other] &&
              (!shared_has[threadIdx.x] ||
               TopKPairBefore(shared_values[other], shared_indices[other],
                              shared_values[threadIdx.x],
                              shared_indices[threadIdx.x],
                              params.largest != 0))) {
            shared_values[threadIdx.x] = shared_values[other];
            shared_indices[threadIdx.x] = shared_indices[other];
            shared_has[threadIdx.x] = true;
          }
        }
        __syncthreads();
      }

      if (threadIdx.x == 0) {
        const int64_t output_index =
            outer_index * params.k * params.inner + k_index * params.inner +
            inner_index;
        values[output_index] = shared_values[0];
        indices[output_index] = shared_indices[0];
        previous = shared_values[0];
        previous_index = shared_indices[0];
        has_previous = true;
      }
      __syncthreads();
    }
  }
}

template <typename T>
musaError_t LaunchTopKTyped(const void* input, void* values, int64_t* indices,
                            MusaTopKParams params, musaStream_t stream) {
  if (params.output_elements == 0) {
    return musaSuccess;
  }
  const int blocks = static_cast<int>(
      params.rows > kMaxBlocks ? kMaxBlocks : params.rows);
  TopKStableKernel<T>
      <<<blocks, kThreadsPerBlock, 0, stream>>>(
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
