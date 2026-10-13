#include "nn/layer_norm_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T, typename ScaleT>
__global__ void LayerNormKernel(const T* input,
                                const ScaleT* scale,
                                const ScaleT* bias,
                                T* output,
                                float* mean,
                                float* inv_std,
                                MusaLayerNormParams params,
                                float epsilon) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= params.rows) {
    return;
  }

  float sum = 0.0f;
  for (int64_t col = threadIdx.x; col < params.norm_size; col += blockDim.x) {
    sum += MusaScalarToFloat(input[row * params.norm_size + col]);
  }

  __shared__ float shared[kThreadsPerBlock];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float row_mean = shared[0] / static_cast<float>(params.norm_size);

  float var_sum = 0.0f;
  for (int64_t col = threadIdx.x; col < params.norm_size; col += blockDim.x) {
    const float value =
        MusaScalarToFloat(input[row * params.norm_size + col]) - row_mean;
    var_sum += value * value;
  }
  shared[threadIdx.x] = var_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float row_inv_std =
      rsqrtf(shared[0] / static_cast<float>(params.norm_size) + epsilon);

  if (threadIdx.x == 0) {
    if (mean != nullptr) {
      mean[row] = row_mean;
    }
    if (inv_std != nullptr) {
      inv_std[row] = row_inv_std;
    }
  }

  for (int64_t col = threadIdx.x; col < params.norm_size; col += blockDim.x) {
    const float normalized =
        (MusaScalarToFloat(input[row * params.norm_size + col]) - row_mean) *
        row_inv_std;
    float value = normalized * MusaScalarToFloat(scale[col]);
    if (params.has_bias) {
      value += MusaScalarToFloat(bias[col]);
    }
    output[row * params.norm_size + col] = MusaScalarFromFloat<T>(value);
  }
}

template <typename T, typename ScaleT>
musaError_t LaunchLayerNormTyped(const void* input,
                                 const void* scale,
                                 const void* bias,
                                 void* output,
                                 float* mean,
                                 float* inv_std,
                                 MusaLayerNormParams params,
                                 float epsilon,
                                 musaStream_t stream) {
  if (params.rows == 0 || params.norm_size == 0) {
    return musaSuccess;
  }
  if (params.norm_size > INT32_MAX) {
    return musaErrorNotSupported;
  }
  LayerNormKernel<T, ScaleT>
      <<<static_cast<int>(params.rows), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const T*>(input),
          reinterpret_cast<const ScaleT*>(scale),
          reinterpret_cast<const ScaleT*>(bias),
          reinterpret_cast<T*>(output), mean, inv_std, params, epsilon);
  return musaGetLastError();
}

template <typename T>
musaError_t DispatchLayerNorm(const void* input, const void* scale,
                              const void* bias, void* output, float* mean,
                              float* inv_std, MusaLayerNormParams params,
                              float epsilon, MusaElementType scale_elem_type,
                              musaStream_t stream) {
  switch (scale_elem_type) {
    case MusaElementType::Float:
      return LaunchLayerNormTyped<T, float>(input, scale, bias, output, mean,
                                            inv_std, params, epsilon, stream);
    case MusaElementType::Float16:
      return LaunchLayerNormTyped<T, __half>(input, scale, bias, output, mean,
                                             inv_std, params, epsilon, stream);
    default:
      return musaErrorNotSupported;
  }
}

}  // namespace

musaError_t LaunchMusaLayerNormKernel(const void* input,
                                      const void* scale,
                                      const void* bias,
                                      void* output,
                                      float* mean,
                                      float* inv_std,
                                      MusaLayerNormParams params,
                                      float epsilon,
                                      MusaElementType elem_type,
                                      MusaElementType scale_elem_type,
                                      musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return DispatchLayerNorm<float>(input, scale, bias, output, mean, inv_std,
                                      params, epsilon, scale_elem_type, stream);
    case MusaElementType::Float16:
      return DispatchLayerNorm<__half>(input, scale, bias, output, mean,
                                       inv_std, params, epsilon,
                                       scale_elem_type, stream);
    default:
      return musaErrorNotSupported;
  }
}
