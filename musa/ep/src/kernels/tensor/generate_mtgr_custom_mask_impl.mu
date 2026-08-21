#include "shared_inc/musa_kernel_common.mu.h"
#include "tensor/generate_mtgr_custom_mask_impl.h"

namespace {

__global__ void GenerateMTGRCustomMaskKernel(
    bool* mask, int64_t* total_length_output, int64_t* target_length_output,
    int64_t upstream_length, int64_t target_length, int64_t total_length,
    int64_t capacity, int64_t element_count) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  if (thread_id == 0) {
    total_length_output[0] = total_length;
    target_length_output[0] = target_length;
  }

  for (int64_t index = thread_id; index < element_count;
       index += total_threads) {
    const int64_t row = index / capacity;
    const int64_t column = index % capacity;
    mask[index] =
        row < total_length && column < total_length &&
        (column < upstream_length || (row == column && row >= upstream_length));
  }
}

__global__ void GenerateMTGRCroppedMaskKernel(
    int32_t* mask, int64_t* target_length_output, int64_t upstream_length,
    int64_t target_length, int64_t dimension, int64_t element_count) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  if (thread_id == 0) {
    target_length_output[0] = target_length;
  }

  for (int64_t index = thread_id; index < element_count;
       index += total_threads) {
    const int64_t row = index / dimension;
    const int64_t column = index % dimension;
    mask[index] =
        column < upstream_length || (row == column && row >= upstream_length);
  }
}

}  // namespace

musaError_t LaunchGenerateMTGRCustomMaskKernel(
    bool* mask, int64_t* total_length_output, int64_t* target_length_output,
    int64_t upstream_length, int64_t target_length, int64_t capacity,
    musaStream_t stream) {
  const int64_t element_count = capacity * capacity;
  const int64_t total_length = upstream_length + target_length;
  GenerateMTGRCustomMaskKernel<<<BlocksForCount(element_count),
                                 kThreadsPerBlock, 0, stream>>>(
      mask, total_length_output, target_length_output, upstream_length,
      target_length, total_length, capacity, element_count);
  return musaGetLastError();
}

musaError_t LaunchGenerateMTGRCroppedMaskKernel(
    int32_t* mask, int64_t* target_length_output, int64_t upstream_length,
    int64_t target_length, int64_t dimension, musaStream_t stream) {
  const int64_t element_count = dimension * dimension;
  GenerateMTGRCroppedMaskKernel<<<BlocksForCount(element_count),
                                  kThreadsPerBlock, 0, stream>>>(
      mask, target_length_output, upstream_length, target_length, dimension,
      element_count);
  return musaGetLastError();
}
