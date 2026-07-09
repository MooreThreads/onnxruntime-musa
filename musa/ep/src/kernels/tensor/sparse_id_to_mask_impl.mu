#include "tensor/sparse_id_to_mask_impl.h"

#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T, typename OutT>
__global__ void SparseIdToMaskKernel(const T* dense_ids, const T* bound_ids,
                                     const T* sparse_ids, OutT* output,
                                     int64_t output_count,
                                     int64_t candidate_count,
                                     int64_t sparse_count, T default_id) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t repeats =
      candidate_count == output_count ? 1 : output_count / candidate_count;
  for (int64_t index = thread_id; index < output_count; index += total_threads) {
    const int64_t candidate_index =
        candidate_count == output_count ? index : index / repeats;
    const int64_t sparse_index = sparse_count == 1 ? 0 : candidate_index;
    const T bound = bound_ids[candidate_index];
    const T sparse_id = sparse_ids[sparse_index];
    const T candidate = bound <= sparse_id ? default_id : bound;
    output[index] = static_cast<OutT>(dense_ids[index] == candidate ? 1 : 0);
  }
}

}  // namespace

musaError_t LaunchMusaSparseIdToMaskInt32Kernel(
    const int32_t* dense_ids, const int32_t* bound_ids,
    const int32_t* sparse_ids, float* output, int64_t output_count,
    int64_t candidate_count, int64_t sparse_count, int32_t default_id,
    musaStream_t stream) {
  if (output_count == 0) {
    return musaSuccess;
  }
  SparseIdToMaskKernel<int32_t, float>
      <<<BlocksForCount(output_count), kThreadsPerBlock, 0, stream>>>(
          dense_ids, bound_ids, sparse_ids, output, output_count,
          candidate_count, sparse_count, default_id);
  return musaGetLastError();
}

musaError_t LaunchMusaSparseIdToMaskInt64Kernel(
    const int64_t* dense_ids, const int64_t* bound_ids,
    const int64_t* sparse_ids, float* output, int64_t output_count,
    int64_t candidate_count, int64_t sparse_count, int64_t default_id,
    musaStream_t stream) {
  if (output_count == 0) {
    return musaSuccess;
  }
  SparseIdToMaskKernel<int64_t, float>
      <<<BlocksForCount(output_count), kThreadsPerBlock, 0, stream>>>(
          dense_ids, bound_ids, sparse_ids, output, output_count,
          candidate_count, sparse_count, default_id);
  return musaGetLastError();
}

musaError_t LaunchMusaSparseIdToMaskInt32Kernel(
    const int32_t* dense_ids, const int32_t* bound_ids,
    const int32_t* sparse_ids, int32_t* output, int64_t output_count,
    int64_t candidate_count, int64_t sparse_count, int32_t default_id,
    musaStream_t stream) {
  if (output_count == 0) {
    return musaSuccess;
  }
  SparseIdToMaskKernel<int32_t, int32_t>
      <<<BlocksForCount(output_count), kThreadsPerBlock, 0, stream>>>(
          dense_ids, bound_ids, sparse_ids, output, output_count,
          candidate_count, sparse_count, default_id);
  return musaGetLastError();
}

musaError_t LaunchMusaSparseIdToMaskInt64Kernel(
    const int64_t* dense_ids, const int64_t* bound_ids,
    const int64_t* sparse_ids, int32_t* output, int64_t output_count,
    int64_t candidate_count, int64_t sparse_count, int64_t default_id,
    musaStream_t stream) {
  if (output_count == 0) {
    return musaSuccess;
  }
  SparseIdToMaskKernel<int64_t, int32_t>
      <<<BlocksForCount(output_count), kThreadsPerBlock, 0, stream>>>(
          dense_ids, bound_ids, sparse_ids, output, output_count,
          candidate_count, sparse_count, default_id);
  return musaGetLastError();
}
