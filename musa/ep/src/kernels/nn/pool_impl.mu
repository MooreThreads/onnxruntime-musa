#include "nn/pool_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T, typename AccT>
__global__ void GlobalAveragePoolKernel(const T* input, T* output,
                                        MusaGlobalAveragePoolParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.output_elements;
       output_index += total_threads) {
    const int64_t batch = output_index / params.channels;
    const int64_t channel = output_index - batch * params.channels;
    const int64_t input_base =
        (batch * params.channels + channel) * params.spatial_elements;

    AccT acc = static_cast<AccT>(0);
    for (int64_t i = 0; i < params.spatial_elements; ++i) {
      acc += static_cast<AccT>(MusaScalarToDouble(input[input_base + i]));
    }
    acc /= static_cast<AccT>(params.spatial_elements);
    output[output_index] = MusaScalarFromDouble<T>(static_cast<double>(acc));
  }
}

template <typename T, typename AccT>
musaError_t LaunchGlobalAveragePoolTyped(const void* input, void* output,
                                         MusaGlobalAveragePoolParams params,
                                         musaStream_t stream) {
  if (params.output_elements == 0) {
    return musaSuccess;
  }
  GlobalAveragePoolKernel<T, AccT>
      <<<BlocksForCount(params.output_elements), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output),
          params);
  return musaGetLastError();
}

template <typename T>
__device__ __forceinline__ bool MusaMaxPoolBetter(T candidate, T current,
                                                  bool has_current) {
  if (!has_current) {
    return true;
  }
  return MusaScalarToDouble(candidate) > MusaScalarToDouble(current);
}

template <typename T>
__global__ void MaxPoolKernel(const T* input, T* output, int64_t* indices,
                              MusaMaxPoolParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.output_elements;
       output_index += total_threads) {
    int64_t remaining = output_index;
    int64_t output_coord[kMusaMaxBroadcastRank];
    for (int32_t dim = 0; dim < params.rank; ++dim) {
      const int64_t stride = params.output_strides[dim];
      output_coord[dim] = stride == 0 ? 0 : remaining / stride;
      remaining -= output_coord[dim] * stride;
    }

    int64_t base_input_offset = 0;
    for (int32_t dim = 0; dim < 2; ++dim) {
      base_input_offset += output_coord[dim] * params.input_strides[dim];
    }

    const int32_t spatial_rank = params.spatial_rank;
    int64_t kernel_elements = 1;
    for (int32_t dim = 0; dim < spatial_rank; ++dim) {
      kernel_elements *= params.kernel_shape[dim];
    }

    bool has_best = false;
    T best{};
    int64_t best_offset = 0;
    for (int64_t kernel_index = 0; kernel_index < kernel_elements;
         ++kernel_index) {
      int64_t kernel_remaining = kernel_index;
      int64_t input_offset = base_input_offset;
      bool in_window = true;
      for (int32_t dim = spatial_rank - 1; dim >= 0; --dim) {
        const int64_t kernel_dim = params.kernel_shape[dim];
        const int64_t k_coord = kernel_remaining % kernel_dim;
        kernel_remaining /= kernel_dim;
        const int32_t full_dim = dim + 2;
        const int64_t in_coord = output_coord[full_dim] * params.strides[dim] -
                                 params.pads_begin[dim] +
                                 k_coord * params.dilations[dim];
        if (in_coord < 0 || in_coord >= params.input_dims[full_dim]) {
          in_window = false;
          break;
        }
        input_offset += in_coord * params.input_strides[full_dim];
      }
      if (!in_window) {
        continue;
      }

      const T candidate = input[input_offset];
      if (MusaMaxPoolBetter<T>(candidate, best, has_best)) {
        has_best = true;
        best = candidate;
        best_offset = input_offset;
      }
    }

    output[output_index] = best;
    if (params.has_indices && indices != nullptr) {
      indices[output_index] = best_offset;
    }
  }
}

template <typename T>
musaError_t LaunchMaxPoolTyped(const void* input, void* output,
                               int64_t* indices, MusaMaxPoolParams params,
                               musaStream_t stream) {
  if (params.output_elements == 0) {
    return musaSuccess;
  }
  MaxPoolKernel<T>
      <<<BlocksForCount(params.output_elements), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output),
          indices, params);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaGlobalAveragePoolKernel(
    const void* input, void* output, MusaGlobalAveragePoolParams params,
    MusaElementType elem_type, musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchGlobalAveragePoolTyped<float, double>(input, output, params,
                                                         stream);
    case MusaElementType::Double:
      return LaunchGlobalAveragePoolTyped<double, double>(input, output, params,
                                                          stream);
    case MusaElementType::Float16:
      return LaunchGlobalAveragePoolTyped<__half, float>(input, output, params,
                                                         stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaMaxPoolKernel(const void* input, void* output,
                                    int64_t* indices, MusaMaxPoolParams params,
                                    MusaElementType elem_type,
                                    musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchMaxPoolTyped<float>(input, output, indices, params, stream);
    case MusaElementType::Double:
      return LaunchMaxPoolTyped<double>(input, output, indices, params, stream);
    case MusaElementType::Float16:
      return LaunchMaxPoolTyped<__half>(input, output, indices, params, stream);
    case MusaElementType::Int8:
      return LaunchMaxPoolTyped<int8_t>(input, output, indices, params, stream);
    case MusaElementType::Uint8:
      return LaunchMaxPoolTyped<uint8_t>(input, output, indices, params,
                                         stream);
    default:
      return musaErrorNotSupported;
  }
}
