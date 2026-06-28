#include "math/mod_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T>
__global__ void ModKernel(const T* lhs, const T* rhs, T* output,
                          MusaBroadcastParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements;
       index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    const T divisor = rhs[rhs_index];
    output[index] = divisor == static_cast<T>(0)
                        ? static_cast<T>(0)
                        : lhs[lhs_index] % divisor;
  }
}

template <typename T>
musaError_t LaunchTypedMod(const void* lhs, const void* rhs, void* output,
                           MusaBroadcastParams params, musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  ModKernel<T><<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0,
                 stream>>>(static_cast<const T*>(lhs),
                           static_cast<const T*>(rhs),
                           static_cast<T*>(output), params);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaModKernel(const void* lhs, const void* rhs,
                                void* output, MusaBroadcastParams params,
                                int32_t elem_type, musaStream_t stream) {
  constexpr int32_t kInt32 = 6;
  constexpr int32_t kInt64 = 7;
  switch (elem_type) {
    case kInt32:
      return LaunchTypedMod<int32_t>(lhs, rhs, output, params, stream);
    case kInt64:
      return LaunchTypedMod<int64_t>(lhs, rhs, output, params, stream);
    default:
      return musaErrorNotSupported;
  }
}
