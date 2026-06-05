#include "nn/pool_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T, typename AccT>
__global__ void GlobalAveragePoolKernel(const T* input,
                                        T* output,
                                        MusaGlobalAveragePoolParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id;
       output_index < params.output_elements; output_index += total_threads) {
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
musaError_t LaunchGlobalAveragePoolTyped(
    const void* input,
    void* output,
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

}  // namespace

musaError_t LaunchMusaGlobalAveragePoolKernel(
    const void* input,
    void* output,
    MusaGlobalAveragePoolParams params,
    MusaElementType elem_type,
    musaStream_t stream) {
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
