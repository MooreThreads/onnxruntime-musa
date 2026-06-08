#include "reduction/masked_gather_reduce_impl.h"

#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void MaskedGatherReduceFloatKernel(const uint8_t* mask,
                                              const float* data,
                                              float* output,
                                              int64_t count,
                                              MusaReduceOp op) {
  float acc = op == MusaReduceOp::Prod ? 1.0f : 0.0f;
  int selected = 0;
  for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask[i] == 0) {
      continue;
    }
    ++selected;
    if (op == MusaReduceOp::Prod) {
      acc *= data[i];
    } else {
      acc += data[i];
    }
  }

  __shared__ float shared_values[kThreadsPerBlock];
  __shared__ int shared_counts[kThreadsPerBlock];
  shared_values[threadIdx.x] = acc;
  shared_counts[threadIdx.x] = selected;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (op == MusaReduceOp::Prod) {
        shared_values[threadIdx.x] *= shared_values[threadIdx.x + stride];
      } else {
        shared_values[threadIdx.x] += shared_values[threadIdx.x + stride];
      }
      shared_counts[threadIdx.x] += shared_counts[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x != 0) {
    return;
  }
  if (op == MusaReduceOp::Mean) {
    output[0] = shared_counts[0] == 0
                    ? NAN
                    : shared_values[0] / static_cast<float>(shared_counts[0]);
  } else {
    output[0] = shared_values[0];
  }
}

}  // namespace

musaError_t LaunchMusaMaskedGatherReduceFloatKernel(const uint8_t* mask,
                                                    const float* data,
                                                    float* output,
                                                    int64_t count,
                                                    MusaReduceOp op,
                                                    musaStream_t stream) {
  if (op != MusaReduceOp::Prod && op != MusaReduceOp::Mean) {
    return musaErrorNotSupported;
  }
  MaskedGatherReduceFloatKernel<<<1, kThreadsPerBlock, 0, stream>>>(
      mask, data, output, count, op);
  return musaGetLastError();
}
