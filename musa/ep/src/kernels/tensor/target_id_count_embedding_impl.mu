#include "shared_inc/musa_kernel_common.mu.h"
#include "tensor/target_id_count_embedding_impl.h"

namespace {

template <typename T>
__device__ int64_t ReadTarget(const void* target, int64_t target_count,
                              int64_t target_elem_size,
                              int64_t target_constant, int64_t row) {
  if (target_count == 0) {
    return target_constant;
  }
  const int64_t index = target_count == 1 ? 0 : row;
  if (target_elem_size == static_cast<int64_t>(sizeof(int32_t))) {
    return static_cast<int64_t>(static_cast<const int32_t*>(target)[index]);
  }
  return static_cast<const int64_t*>(target)[index];
}

template <typename T>
__global__ void TargetIdCountEmbeddingKernel(
    const T* __restrict__ ids, const void* __restrict__ target,
    const float* __restrict__ table, float* __restrict__ embedding_output,
    float* __restrict__ count_output, int64_t rows, int64_t sequence_length,
    int64_t embedding_dim, int64_t table_rows, int64_t target_count,
    int64_t target_elem_size, int64_t target_constant, int64_t pad,
    int64_t cap) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ int64_t counts[kThreadsPerBlock];
  const int64_t target_value = ReadTarget<T>(
      target, target_count, target_elem_size, target_constant, row);
  int64_t local_count = 0;
  const int64_t row_offset = row * sequence_length;
  for (int64_t col = threadIdx.x; col < sequence_length; col += blockDim.x) {
    const int64_t id = static_cast<int64_t>(ids[row_offset + col]);
    if (id != pad && id == target_value) {
      ++local_count;
    }
  }

  counts[threadIdx.x] = local_count;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      counts[threadIdx.x] += counts[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const int64_t count = counts[0];
  const int64_t bucket = min(max(count, static_cast<int64_t>(0)),
                             min(cap, table_rows - 1));
  if (threadIdx.x == 0) {
    count_output[row] = static_cast<float>(count);
  }
  for (int64_t i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
    embedding_output[row * embedding_dim + i] =
        table[bucket * embedding_dim + i];
  }
}

template <typename T>
musaError_t LaunchTargetIdCountEmbeddingKernel(
    const T* ids, const void* target, const float* table,
    float* embedding_output, float* count_output, int64_t rows,
    int64_t sequence_length, int64_t embedding_dim, int64_t table_rows,
    int64_t target_count, int64_t target_elem_size, int64_t target_constant,
    int64_t pad, int64_t cap, musaStream_t stream) {
  if (rows == 0 || sequence_length == 0 || embedding_dim == 0) {
    return musaSuccess;
  }
  TargetIdCountEmbeddingKernel<T>
      <<<static_cast<unsigned int>(rows), kThreadsPerBlock, 0, stream>>>(
          ids, target, table, embedding_output, count_output, rows,
          sequence_length, embedding_dim, table_rows, target_count,
          target_elem_size, target_constant, pad, cap);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaTargetIdCountEmbeddingInt32Kernel(
    const int32_t* ids, const void* target, const float* table,
    float* embedding_output, float* count_output, int64_t rows,
    int64_t sequence_length, int64_t embedding_dim, int64_t table_rows,
    int64_t target_count, int64_t target_elem_size, int64_t target_constant,
    int64_t pad, int64_t cap, musaStream_t stream) {
  return LaunchTargetIdCountEmbeddingKernel(
      ids, target, table, embedding_output, count_output, rows,
      sequence_length, embedding_dim, table_rows, target_count,
      target_elem_size, target_constant, pad, cap, stream);
}

musaError_t LaunchMusaTargetIdCountEmbeddingInt64Kernel(
    const int64_t* ids, const void* target, const float* table,
    float* embedding_output, float* count_output, int64_t rows,
    int64_t sequence_length, int64_t embedding_dim, int64_t table_rows,
    int64_t target_count, int64_t target_elem_size, int64_t target_constant,
    int64_t pad, int64_t cap, musaStream_t stream) {
  return LaunchTargetIdCountEmbeddingKernel(
      ids, target, table, embedding_output, count_output, rows,
      sequence_length, embedding_dim, table_rows, target_count,
      target_elem_size, target_constant, pad, cap, stream);
}
