#include "nn/batch_norm_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T>
__global__ void BatchNormalizationKernel(const T* input,
                                         const T* scale,
                                         const T* bias,
                                         const T* mean,
                                         const T* variance,
                                         T* output,
                                         MusaBatchNormParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements;
       index += total_threads) {
    const int64_t channel =
        params.channels == 1 ? 0 : (index / params.spatial_size) % params.channels;
    const double value =
        (MusaScalarToDouble(input[index]) - MusaScalarToDouble(mean[channel])) *
            (1.0 / sqrt(MusaScalarToDouble(variance[channel]) +
                         static_cast<double>(params.epsilon))) *
            MusaScalarToDouble(scale[channel]) +
        MusaScalarToDouble(bias[channel]);
    output[index] = MusaScalarFromDouble<T>(value);
  }
}

template <typename T>
musaError_t LaunchBatchNormalizationTyped(const void* input,
                                          const void* scale,
                                          const void* bias,
                                          const void* mean,
                                          const void* variance,
                                          void* output,
                                          MusaBatchNormParams params,
                                          musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  BatchNormalizationKernel<T>
      <<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const T*>(input), reinterpret_cast<const T*>(scale),
          reinterpret_cast<const T*>(bias), reinterpret_cast<const T*>(mean),
          reinterpret_cast<const T*>(variance), reinterpret_cast<T*>(output),
          params);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaBatchNormalizationKernel(const void* input,
                                               const void* scale,
                                               const void* bias,
                                               const void* mean,
                                               const void* variance,
                                               void* output,
                                               MusaBatchNormParams params,
                                               MusaElementType elem_type,
                                               musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchBatchNormalizationTyped<float>(
          input, scale, bias, mean, variance, output, params, stream);
    case MusaElementType::Double:
      return LaunchBatchNormalizationTyped<double>(
          input, scale, bias, mean, variance, output, params, stream);
    case MusaElementType::Float16:
      return LaunchBatchNormalizationTyped<__half>(
          input, scale, bias, mean, variance, output, params, stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaBatchNormalizationFloatKernel(const float* input,
                                                    const float* scale,
                                                    const float* bias,
                                                    const float* mean,
                                                    const float* variance,
                                                    float* output,
                                                    MusaBatchNormParams params,
                                                    musaStream_t stream) {
  return LaunchMusaBatchNormalizationKernel(
      input, scale, bias, mean, variance, output, params,
      MusaElementType::Float, stream);
}
