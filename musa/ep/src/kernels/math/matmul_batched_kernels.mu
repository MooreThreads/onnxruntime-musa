#include "matmul_batched_kernels.h"

#include <stdint.h>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 4096;

int BlocksForCount(int64_t count) {
  int64_t blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (blocks > kMaxBlocks) {
    blocks = kMaxBlocks;
  }
  return static_cast<int>(blocks);
}

__global__ void BatchedMatMulFloatKernel(const float* a,
                                         const float* b,
                                         float* output,
                                         MusaBatchedMatMulParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.total_elements;
       output_index += total_threads) {
    int64_t remaining = output_index;
    int64_t a_base = 0;
    int64_t b_base = 0;
    int64_t row = 0;
    int64_t col = 0;
    for (int32_t dim = 0; dim < params.output_rank; ++dim) {
      const int64_t coord = remaining / params.output_strides[dim];
      remaining -= coord * params.output_strides[dim];
      if (dim < params.batch_rank) {
        a_base += coord * params.a_batch_strides[dim];
        b_base += coord * params.b_batch_strides[dim];
      } else if (dim == params.batch_rank) {
        row = coord;
      } else {
        col = coord;
      }
    }

    float sum = 0.0f;
    for (int64_t kk = 0; kk < params.k; ++kk) {
      sum += a[a_base + row * params.a_row_stride + kk * params.a_col_stride] *
             b[b_base + kk * params.b_row_stride + col * params.b_col_stride];
    }
    output[output_index] = sum;
  }
}

}  // namespace

musaError_t LaunchMusaBatchedMatMulFloatKernel(
    const float* a, const float* b, float* output,
    MusaBatchedMatMulParams params, musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  BatchedMatMulFloatKernel<<<BlocksForCount(params.total_elements),
                             kThreadsPerBlock, 0, stream>>>(a, b, output,
                                                             params);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}
