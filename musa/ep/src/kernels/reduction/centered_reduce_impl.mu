#include "reduction/centered_reduce_impl.h"

#include "shared_inc/musa_kernel_common.mu.h"

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float ReduceIdentity(MusaReduceOp op) {
  return op == MusaReduceOp::Prod ? 1.0f : 0.0f;
}

__device__ __forceinline__ float ReduceUpdate(float acc, float value,
                                              MusaReduceOp op) {
  return op == MusaReduceOp::Prod ? acc * value : acc + value;
}

__device__ __forceinline__ float BlockReduce(float value, MusaReduceOp op) {
  __shared__ float shared[kThreadsPerBlock / kWarpSize];
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x >> 5;

#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    value = ReduceUpdate(value, __shfl_xor_sync(0xffffffff, value, mask), op);
  }

  if (lane == 0) {
    shared[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    value = threadIdx.x < (kThreadsPerBlock / kWarpSize)
                ? shared[threadIdx.x]
                : ReduceIdentity(op);
#pragma unroll
    for (int mask = (kThreadsPerBlock / kWarpSize) / 2; mask > 0;
         mask >>= 1) {
      value = ReduceUpdate(value, __shfl_xor_sync(0xffffffff, value, mask), op);
    }
    if (lane == 0) {
      shared[0] = value;
    }
  }
  __syncthreads();
  return shared[0];
}

__global__ void CenteredReduceFloatKernel(const float* input,
                                          float* first_reduce,
                                          float* second_reduce, int64_t inner,
                                          MusaReduceOp first_op,
                                          MusaReduceOp second_op) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const float* row_input = input + row * inner;

  float first_acc = ReduceIdentity(first_op);
  for (int64_t col = threadIdx.x; col < inner; col += blockDim.x) {
    first_acc = ReduceUpdate(first_acc, row_input[col], first_op);
  }
  const float first = BlockReduce(first_acc, first_op);

  float second_acc = ReduceIdentity(second_op);
  for (int64_t col = threadIdx.x; col < inner; col += blockDim.x) {
    const float centered = row_input[col] - first;
    second_acc = ReduceUpdate(second_acc, centered * centered, second_op);
  }
  const float second = BlockReduce(second_acc, second_op);

  if (threadIdx.x == 0) {
    first_reduce[row] = first;
    second_reduce[row] = second;
  }
}

}  // namespace

musaError_t LaunchMusaCenteredReduceFloatKernel(
    const float* input, float* first_reduce, float* second_reduce,
    int64_t rows, int64_t inner, MusaReduceOp first_op,
    MusaReduceOp second_op, musaStream_t stream) {
  if (rows == 0 || inner == 0) {
    return musaSuccess;
  }
  if ((first_op != MusaReduceOp::Prod && first_op != MusaReduceOp::Sum) ||
      (second_op != MusaReduceOp::Prod && second_op != MusaReduceOp::Sum)) {
    return musaErrorNotSupported;
  }
  CenteredReduceFloatKernel<<<static_cast<int>(rows), kThreadsPerBlock, 0,
                              stream>>>(input, first_reduce, second_reduce,
                                        inner, first_op, second_op);
  return musaGetLastError();
}
