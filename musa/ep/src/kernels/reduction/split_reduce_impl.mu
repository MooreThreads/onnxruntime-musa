#include "reduction/split_reduce_impl.h"

#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ float ReduceSegment(const float* input,
                                               int64_t base, int64_t inner,
                                               int64_t width,
                                               MusaSplitReduceMode mode) {
  float acc = mode == MusaSplitReduceMode::Prod ? 1.0f : 0.0f;
  for (int64_t i = 0; i < width; ++i) {
    const float value = input[base + i * inner];
    if (mode == MusaSplitReduceMode::Prod) {
      acc *= value;
    } else {
      acc += value;
    }
  }
  if (mode == MusaSplitReduceMode::Mean) {
    acc /= static_cast<float>(width);
  }
  return acc;
}

__global__ void SplitReduce2FloatKernel(
    const float* input, float* output0, float* output1, int64_t batch,
    int64_t axis_dim, int64_t inner, int64_t offset0, int64_t width0,
    MusaSplitReduceMode mode0, int64_t offset1, int64_t width1,
    MusaSplitReduceMode mode1) {
  const int64_t total = batch * inner;
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < total; index += total_threads) {
    const int64_t row = index / inner;
    const int64_t col = index - row * inner;
    const int64_t row_base = row * axis_dim * inner + col;
    output0[index] =
        ReduceSegment(input, row_base + offset0 * inner, inner, width0, mode0);
    output1[index] =
        ReduceSegment(input, row_base + offset1 * inner, inner, width1, mode1);
  }
}

}  // namespace

musaError_t LaunchMusaSplitReduce2Float(
    const float* input, float* output0, float* output1, int64_t batch,
    int64_t axis_dim, int64_t inner, int64_t offset0, int64_t width0,
    MusaSplitReduceMode mode0, int64_t offset1, int64_t width1,
    MusaSplitReduceMode mode1, musaStream_t stream) {
  if (batch == 0 || inner == 0) {
    return musaSuccess;
  }
  SplitReduce2FloatKernel<<<BlocksForCount(batch * inner), kThreadsPerBlock, 0,
                             stream>>>(input, output0, output1, batch,
                                       axis_dim, inner, offset0, width0, mode0,
                                       offset1, width1, mode1);
  return musaGetLastError();
}
