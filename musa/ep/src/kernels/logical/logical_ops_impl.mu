#include "logical/logical_ops_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void NotBoolKernel(const uint8_t* input,
                              uint8_t* output,
                              int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = static_cast<uint8_t>(!input[index]);
  }
}

__global__ void OrBoolKernel(const uint8_t* lhs,
                             const uint8_t* rhs,
                             uint8_t* output,
                             MusaBroadcastParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    output[index] = static_cast<uint8_t>(lhs[lhs_index] || rhs[rhs_index]);
  }
}

__global__ void AndBoolKernel(const uint8_t* lhs,
                              const uint8_t* rhs,
                              uint8_t* output,
                              MusaBroadcastParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    output[index] = static_cast<uint8_t>(lhs[lhs_index] && rhs[rhs_index]);
  }
}

template <typename T>
__global__ void BitwiseAndKernel(const T* lhs,
                                 const T* rhs,
                                 T* output,
                                 MusaBroadcastParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    output[index] = static_cast<T>(lhs[lhs_index] & rhs[rhs_index]);
  }
}

}  // namespace

musaError_t LaunchMusaNotBoolKernel(const uint8_t* input,
                                    uint8_t* output,
                                    int64_t count,
                                    musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  NotBoolKernel<<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(input, output, count);
  return musaGetLastError();
}

musaError_t LaunchMusaOrBoolKernel(const uint8_t* lhs,
                                   const uint8_t* rhs,
                                   uint8_t* output,
                                   MusaBroadcastParams params,
                                   musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  OrBoolKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(lhs, rhs, output, params);
  return musaGetLastError();
}

musaError_t LaunchMusaAndBoolKernel(const uint8_t* lhs,
                                    const uint8_t* rhs,
                                    uint8_t* output,
                                    MusaBroadcastParams params,
                                    musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  AndBoolKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(lhs, rhs, output, params);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchBitwiseAndTyped(const void* lhs,
                                  const void* rhs,
                                  void* output,
                                  MusaBroadcastParams params,
                                  musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  BitwiseAndKernel<T><<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(lhs), reinterpret_cast<const T*>(rhs),
      reinterpret_cast<T*>(output), params);
  return musaGetLastError();
}

musaError_t LaunchMusaBitwiseAndKernel(const void* lhs,
                                       const void* rhs,
                                       void* output,
                                       MusaBroadcastParams params,
                                       MusaElementType elem_type,
                                       musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Uint8:
      return LaunchBitwiseAndTyped<uint8_t>(lhs, rhs, output, params, stream);
    case MusaElementType::Uint16:
      return LaunchBitwiseAndTyped<uint16_t>(lhs, rhs, output, params, stream);
    case MusaElementType::Uint32:
      return LaunchBitwiseAndTyped<uint32_t>(lhs, rhs, output, params, stream);
    case MusaElementType::Uint64:
      return LaunchBitwiseAndTyped<uint64_t>(lhs, rhs, output, params, stream);
    case MusaElementType::Int8:
      return LaunchBitwiseAndTyped<int8_t>(lhs, rhs, output, params, stream);
    case MusaElementType::Int16:
      return LaunchBitwiseAndTyped<int16_t>(lhs, rhs, output, params, stream);
    case MusaElementType::Int32:
      return LaunchBitwiseAndTyped<int32_t>(lhs, rhs, output, params, stream);
    case MusaElementType::Int64:
      return LaunchBitwiseAndTyped<int64_t>(lhs, rhs, output, params, stream);
    default:
      return musaErrorNotSupported;
  }
}
