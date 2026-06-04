#include "generator/random_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ uint32_t RandomHash(uint64_t index,
                                               uint64_t seed) {
  uint64_t x = index + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return static_cast<uint32_t>(x >> 32);
}

__device__ __forceinline__ float RandomUniformValue(int64_t index,
                                                    float low,
                                                    float high,
                                                    uint64_t seed) {
  const uint32_t bits = RandomHash(static_cast<uint64_t>(index), seed);
  const float unit =
      static_cast<float>(bits & 0x00ffffffU) * (1.0f / 16777216.0f);
  return low + (high - low) * unit;
}

template <typename T>
__global__ void RandomUniformKernel(T* output,
                                    int64_t count,
                                    float low,
                                    float high,
                                    uint64_t seed) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = MusaScalarFromFloat<T>(
        RandomUniformValue(index, low, high, seed));
  }
}

template <typename T>
musaError_t LaunchRandomUniformTyped(void* output,
                                     int64_t count,
                                     float low,
                                     float high,
                                     uint64_t seed,
                                     musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  RandomUniformKernel<T><<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<T*>(output), count, low, high, seed);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaRandomUniformKernel(void* output,
                                          int64_t count,
                                          float low,
                                          float high,
                                          uint64_t seed,
                                          MusaElementType elem_type,
                                          musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchRandomUniformTyped<float>(output, count, low, high, seed,
                                             stream);
    case MusaElementType::Double:
      return LaunchRandomUniformTyped<double>(output, count, low, high, seed,
                                              stream);
    case MusaElementType::Float16:
      return LaunchRandomUniformTyped<__half>(output, count, low, high, seed,
                                              stream);
    case MusaElementType::BFloat16:
      return LaunchRandomUniformTyped<__mt_bfloat16>(output, count, low, high,
                                                     seed, stream);
    default:
      return musaErrorNotSupported;
  }
}
