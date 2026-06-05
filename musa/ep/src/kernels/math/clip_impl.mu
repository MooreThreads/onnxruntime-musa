#include <limits>

#include "math/clip_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T>
__device__ __forceinline__ T ClipDefaultMin() {
  return ::std::numeric_limits<T>::lowest();
}

template <typename T>
__device__ __forceinline__ T ClipDefaultMax() {
  return ::std::numeric_limits<T>::max();
}

template <typename T>
__global__ void ClipKernel(const T* input, T* output, MusaClipParams params) {
  const T min_value = params.has_min
                          ? reinterpret_cast<const T*>(params.min_data)[0]
                          : ClipDefaultMin<T>();
  const T max_value = params.has_max
                          ? reinterpret_cast<const T*>(params.max_data)[0]
                          : ClipDefaultMax<T>();
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.count;
       index += total_threads) {
    T value = input[index];
    if (params.has_min && value < min_value) {
      value = min_value;
    }
    if (params.has_max && value > max_value) {
      value = max_value;
    }
    output[index] = value;
  }
}

template <typename T>
__global__ void ClipFloatLikeKernel(const T* input, T* output,
                                    MusaClipParams params) {
  const float min_value =
      params.has_min
          ? MusaScalarToFloat(reinterpret_cast<const T*>(params.min_data)[0])
          : -INFINITY;
  const float max_value =
      params.has_max
          ? MusaScalarToFloat(reinterpret_cast<const T*>(params.max_data)[0])
          : INFINITY;
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.count;
       index += total_threads) {
    float value = MusaScalarToFloat(input[index]);
    if (params.has_min && value < min_value) {
      value = min_value;
    }
    if (params.has_max && value > max_value) {
      value = max_value;
    }
    output[index] = MusaScalarFromFloat<T>(value);
  }
}

template <typename T>
musaError_t LaunchClipTyped(const void* input, void* output,
                            MusaClipParams params, musaStream_t stream) {
  if (params.count == 0) {
    return musaSuccess;
  }
  ClipKernel<T><<<BlocksForCount(params.count), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), params);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchClipFloatLikeTyped(const void* input, void* output,
                                     MusaClipParams params,
                                     musaStream_t stream) {
  if (params.count == 0) {
    return musaSuccess;
  }
  ClipFloatLikeKernel<T>
      <<<BlocksForCount(params.count), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output),
          params);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaClipKernel(const void* input, void* output,
                                 MusaClipParams params,
                                 MusaElementType elem_type,
                                 musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchClipTyped<float>(input, output, params, stream);
    case MusaElementType::Double:
      return LaunchClipTyped<double>(input, output, params, stream);
    case MusaElementType::Int8:
      return LaunchClipTyped<int8_t>(input, output, params, stream);
    case MusaElementType::Uint8:
      return LaunchClipTyped<uint8_t>(input, output, params, stream);
    case MusaElementType::Int64:
      return LaunchClipTyped<int64_t>(input, output, params, stream);
    case MusaElementType::Uint64:
      return LaunchClipTyped<uint64_t>(input, output, params, stream);
    case MusaElementType::Float16:
      return LaunchClipFloatLikeTyped<__half>(input, output, params, stream);
    default:
      return musaErrorNotSupported;
  }
}
