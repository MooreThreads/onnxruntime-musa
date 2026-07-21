#include "kernels/tensor/segment_max_broadcast_impl.h"

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void SegmentMaxBroadcastKernel(const int64_t* segment_ids,
                                          const float* values, float* output,
                                          int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  const int64_t segment = segment_ids[index];
  float maximum = values[index];
  for (int64_t candidate = 0; candidate < count; ++candidate) {
    if (segment_ids[candidate] == segment && values[candidate] > maximum) {
      maximum = values[candidate];
    }
  }
  output[index] = maximum;
}

}  // namespace

musaError_t LaunchMusaSegmentMaxBroadcastKernel(
    const int64_t* segment_ids, const float* values, float* output,
    int64_t count, musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  const int blocks = static_cast<int>((count + kThreadsPerBlock - 1) /
                                      kThreadsPerBlock);
  SegmentMaxBroadcastKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      segment_ids, values, output, count);
  return musaGetLastError();
}
