#include "generator/range_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T>
__global__ void RangeKernel(T* output,
                            MusaRangeParams params,
                            double start,
                            double delta) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.count;
       index += total_threads) {
    output[index] = static_cast<T>(start + delta * static_cast<double>(index));
  }
}

template <typename T>
musaError_t LaunchRangeTyped(void* output,
                             MusaRangeParams params,
                             double start,
                             double delta,
                             musaStream_t stream) {
  if (params.count == 0) {
    return musaSuccess;
  }
  RangeKernel<T><<<BlocksForCount(params.count), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<T*>(output), params, start, delta);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaRangeKernel(void* output,
                                  MusaRangeParams params,
                                  MusaElementType elem_type,
                                  double start,
                                  double delta,
                                  musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Int16:
      return LaunchRangeTyped<int16_t>(output, params, start, delta, stream);
    case MusaElementType::Int32:
      return LaunchRangeTyped<int32_t>(output, params, start, delta, stream);
    case MusaElementType::Int64:
      return LaunchRangeTyped<int64_t>(output, params, start, delta, stream);
    case MusaElementType::Float:
      return LaunchRangeTyped<float>(output, params, start, delta, stream);
    case MusaElementType::Double:
      return LaunchRangeTyped<double>(output, params, start, delta, stream);
    default:
      return musaErrorNotSupported;
  }
}
