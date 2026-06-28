#include "activation/prelu_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T>
__global__ void PReluKernel(const T* input, const T* slope, T* output,
                            MusaBroadcastParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements;
       index += total_threads) {
    int64_t input_index = 0;
    int64_t slope_index = 0;
    ResolveBroadcastIndices(index, params, input_index, slope_index);
    const T x = input[input_index];
    output[index] = x >= static_cast<T>(0) ? x : x * slope[slope_index];
  }
}

template <typename T>
__global__ void PReluFloatLikeKernel(const T* input, const T* slope, T* output,
                                     MusaBroadcastParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements;
       index += total_threads) {
    int64_t input_index = 0;
    int64_t slope_index = 0;
    ResolveBroadcastIndices(index, params, input_index, slope_index);
    const float x = MusaScalarToFloat(input[input_index]);
    const float alpha = MusaScalarToFloat(slope[slope_index]);
    output[index] = MusaScalarFromFloat<T>(x >= 0.0f ? x : x * alpha);
  }
}

template <typename T>
musaError_t LaunchTypedPRelu(const void* input, const void* slope,
                             void* output, MusaBroadcastParams params,
                             musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  PReluKernel<T><<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0,
                   stream>>>(static_cast<const T*>(input),
                             static_cast<const T*>(slope),
                             static_cast<T*>(output), params);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchFloatLikePRelu(const void* input, const void* slope,
                                 void* output, MusaBroadcastParams params,
                                 musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  PReluFloatLikeKernel<T>
      <<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
          static_cast<const T*>(input), static_cast<const T*>(slope),
          static_cast<T*>(output), params);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaPReluKernel(const void* input, const void* slope,
                                  void* output,
                                  MusaBroadcastParams params,
                                  int32_t elem_type, musaStream_t stream) {
  constexpr int32_t kFloat = 1;
  constexpr int32_t kFloat16 = 10;
  constexpr int32_t kDouble = 11;
  constexpr int32_t kBFloat16 = 16;
  switch (elem_type) {
    case kFloat:
      return LaunchTypedPRelu<float>(input, slope, output, params, stream);
    case kDouble:
      return LaunchTypedPRelu<double>(input, slope, output, params, stream);
    case kFloat16:
      return LaunchFloatLikePRelu<__half>(input, slope, output, params,
                                          stream);
    case kBFloat16:
      return LaunchFloatLikePRelu<__mt_bfloat16>(input, slope, output, params,
                                                 stream);
    default:
      return musaErrorNotSupported;
  }
}
