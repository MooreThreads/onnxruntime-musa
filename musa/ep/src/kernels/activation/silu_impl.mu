#include "activation/silu_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ float SiluValue(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ double SiluValue(double x) {
  return x / (1.0 + exp(-x));
}

template <typename T>
__global__ void SiluKernel(const T* input, T* output, int64_t count) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    const double value = MusaScalarToDouble(input[index]);
    output[index] = MusaScalarFromDouble<T>(SiluValue(value));
  }
}

__global__ void SiluFloatKernel(const float* input, float* output,
                                int64_t count) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = SiluValue(input[index]);
  }
}

template <typename T>
musaError_t LaunchSiluTyped(const void* input, void* output, int64_t count,
                            musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  SiluKernel<T><<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), count);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaSiluKernel(const void* input, void* output, int64_t count,
                                 MusaElementType elem_type,
                                 musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      if (count == 0) {
        return musaSuccess;
      }
      SiluFloatKernel<<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const float*>(input),
          reinterpret_cast<float*>(output), count);
      return musaGetLastError();
    case MusaElementType::Double:
      return LaunchSiluTyped<double>(input, output, count, stream);
    case MusaElementType::Float16:
      return LaunchSiluTyped<__half>(input, output, count, stream);
    case MusaElementType::BFloat16:
      return LaunchSiluTyped<__mt_bfloat16>(input, output, count, stream);
    default:
      return musaErrorNotSupported;
  }
}
