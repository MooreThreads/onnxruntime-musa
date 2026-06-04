#include "nn/batch_norm_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void BatchNormalizationFloatKernel(const float* input,
                                              const float* scale,
                                              const float* bias,
                                              const float* mean,
                                              const float* variance,
                                              float* output,
                                              MusaBatchNormParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    const int64_t channel = params.channels == 1 ? 0 : (index / params.spatial_size) % params.channels;
    output[index] = (input[index] - mean[channel]) * rsqrtf(variance[channel] + params.epsilon) *
                    scale[channel] + bias[channel];
  }
}

}  // namespace

musaError_t LaunchMusaBatchNormalizationFloatKernel(const float* input,
                                                    const float* scale,
                                                    const float* bias,
                                                    const float* mean,
                                                    const float* variance,
                                                    float* output,
                                                    MusaBatchNormParams params,
                                                    musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  BatchNormalizationFloatKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
      input, scale, bias, mean, variance, output, params);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}
