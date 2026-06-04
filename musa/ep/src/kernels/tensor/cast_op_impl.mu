#include "tensor/cast_op_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void CastInt32ToFloatKernel(const int32_t* input,
                                       float* output,
                                       int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = static_cast<float>(input[index]);
  }
}

__global__ void CastInt64ToFloatKernel(const int64_t* input,
                                       float* output,
                                       int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = static_cast<float>(input[index]);
  }
}

}  // namespace

musaError_t LaunchMusaCastInt32ToFloatKernel(const int32_t* input,
                                             float* output,
                                             int64_t count,
                                             musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  CastInt32ToFloatKernel<<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(input, output, count);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}

musaError_t LaunchMusaCastInt64ToFloatKernel(const int64_t* input,
                                             float* output,
                                             int64_t count,
                                             musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  CastInt64ToFloatKernel<<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(input, output, count);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}
