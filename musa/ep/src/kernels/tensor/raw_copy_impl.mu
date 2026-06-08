#include "tensor/raw_copy_impl.h"

#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void RawCopy3FloatKernel(
    const float* input0, const float* input1, const float* input2,
    float* output0, float* output1, float* output2, int64_t element_count) {
  const int64_t total = element_count * 3;
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t linear = thread_id; linear < total; linear += total_threads) {
    const int64_t tensor = linear / element_count;
    const int64_t offset = linear - tensor * element_count;
    if (tensor == 0) {
      output0[offset] = input0[offset];
    } else if (tensor == 1) {
      output1[offset] = input1[offset];
    } else {
      output2[offset] = input2[offset];
    }
  }
}

}  // namespace

musaError_t LaunchMusaRawCopy3Float(
    const float* input0, const float* input1, const float* input2,
    float* output0, float* output1, float* output2, int64_t element_count,
    musaStream_t stream) {
  if (element_count == 0) {
    return musaSuccess;
  }
  RawCopy3FloatKernel<<<BlocksForCount(element_count * 3), kThreadsPerBlock, 0,
                        stream>>>(input0, input1, input2, output0, output1,
                                  output2, element_count);
  return musaGetLastError();
}
