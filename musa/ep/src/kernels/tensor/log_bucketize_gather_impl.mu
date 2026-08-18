#include <math.h>

#include "shared_inc/musa_kernel_common.mu.h"
#include "tensor/log_bucketize_gather_impl.h"

namespace {

__device__ __forceinline__ int64_t ReadInteger(const void* data,
                                               int32_t element_size,
                                               int64_t offset) {
  if (element_size == 4) {
    return static_cast<int64_t>(reinterpret_cast<const int32_t*>(data)[offset]);
  }
  if (element_size == 8) {
    return reinterpret_cast<const int64_t*>(data)[offset];
  }
  return 0;
}

__device__ __forceinline__ int64_t ClampIndex(int64_t value, int64_t min_value,
                                              int64_t max_value) {
  if (value < min_value) {
    return min_value;
  }
  if (value > max_value) {
    return max_value;
  }
  return value;
}

__device__ __forceinline__ int64_t ComputeGatherIndex(
    const void* current, const void* sequence, int32_t current_element_size,
    int32_t sequence_element_size, int64_t current_count,
    int64_t sequence_index, int64_t table_rows, float diff_min,
    float time_scale, float value_min, float log_base, float bucket_start,
    float bucket_span, float bucket_scale, int64_t bucket_offset,
    int64_t bucket_min, int64_t bucket_max, float lower_threshold,
    float upper_threshold) {
  const int64_t current_index =
      current_count == 1 ? 0 : sequence_index % current_count;
  const int64_t current_value =
      ReadInteger(current, current_element_size, current_index);
  const int64_t sequence_value =
      ReadInteger(sequence, sequence_element_size, sequence_index);

  float diff = static_cast<float>(current_value - sequence_value);
  if (diff < diff_min) {
    diff = diff_min;
  }
  float scaled_value = diff / time_scale;
  if (scaled_value < value_min) {
    scaled_value = value_min;
  }
  const float log_bucket = logf(scaled_value) / log_base;
  float mapped =
      floorf(((log_bucket - bucket_start) / bucket_span) * bucket_scale);
  int64_t gather_index =
      static_cast<int64_t>(mapped) + static_cast<int64_t>(bucket_offset);
  gather_index = ClampIndex(gather_index, bucket_min, bucket_max);
  if (log_bucket >= upper_threshold) {
    gather_index = bucket_max;
  }
  if (log_bucket <= lower_threshold) {
    gather_index = 0;
  }
  if (gather_index < 0 || gather_index >= table_rows) {
    gather_index = 0;
  }
  return gather_index;
}

struct alignas(16) Copy16 {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t w;
};

__global__ void LogBucketizeGatherElement2Block128Kernel(
    const void* table, const void* current, const void* sequence, void* output,
    int32_t current_element_size, int32_t sequence_element_size,
    int64_t current_count, int64_t sequence_count, int64_t table_rows,
    float diff_min, float time_scale, float value_min, float log_base,
    float bucket_start, float bucket_span, float bucket_scale,
    int64_t bucket_offset, int64_t bucket_min, int64_t bucket_max,
    float lower_threshold, float upper_threshold) {
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kElementSize = 2;
  constexpr int kRowsPerBlock = 8;
  constexpr int kCopyBytes = 16;
  constexpr int kChunksPerRow =
      static_cast<int>((kBlockSize * kElementSize) / kCopyBytes);

  __shared__ int64_t gather_indices[kRowsPerBlock];

  if (threadIdx.x < kRowsPerBlock) {
    const int64_t sequence_index =
        static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + threadIdx.x;
    if (sequence_index < sequence_count) {
      gather_indices[threadIdx.x] = ComputeGatherIndex(
          current, sequence, current_element_size, sequence_element_size,
          current_count, sequence_index, table_rows, diff_min, time_scale,
          value_min, log_base, bucket_start, bucket_span, bucket_scale,
          bucket_offset, bucket_min, bucket_max, lower_threshold,
          upper_threshold);
    }
  }
  __syncthreads();

  const int row_in_block = threadIdx.x / kChunksPerRow;
  const int chunk_in_row = threadIdx.x - row_in_block * kChunksPerRow;
  const int64_t sequence_index =
      static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + row_in_block;
  if (row_in_block >= kRowsPerBlock || sequence_index >= sequence_count) {
    return;
  }

  const uint8_t* src_base = reinterpret_cast<const uint8_t*>(table) +
                            gather_indices[row_in_block] * kBlockSize *
                                kElementSize;
  uint8_t* dst_base = reinterpret_cast<uint8_t*>(output) +
                      sequence_index * kBlockSize * kElementSize;
  reinterpret_cast<Copy16*>(dst_base)[chunk_in_row] =
      reinterpret_cast<const Copy16*>(src_base)[chunk_in_row];
}

__global__ void LogBucketizeGatherKernel(
    const void* table, const void* current, const void* sequence, void* output,
    int32_t element_size, int32_t current_element_size,
    int32_t sequence_element_size, int64_t current_count,
    int64_t sequence_count, int64_t table_rows, int64_t block_size,
    float diff_min, float time_scale, float value_min, float log_base,
    float bucket_start, float bucket_span, float bucket_scale,
    int64_t bucket_offset, int64_t bucket_min, int64_t bucket_max,
    float lower_threshold, float upper_threshold, int64_t output_count) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < output_count;
       output_index += total_threads) {
    const int64_t sequence_index = output_index / block_size;
    const int64_t element_offset = output_index % block_size;
    const int64_t gather_index = ComputeGatherIndex(
        current, sequence, current_element_size, sequence_element_size,
        current_count, sequence_index, table_rows, diff_min, time_scale,
        value_min, log_base, bucket_start, bucket_span, bucket_scale,
        bucket_offset, bucket_min, bucket_max, lower_threshold,
        upper_threshold);

    const int64_t table_index = gather_index * block_size + element_offset;
    if (element_size == 4) {
      reinterpret_cast<uint32_t*>(output)[output_index] =
          reinterpret_cast<const uint32_t*>(table)[table_index];
    } else if (element_size == 8) {
      reinterpret_cast<uint64_t*>(output)[output_index] =
          reinterpret_cast<const uint64_t*>(table)[table_index];
    } else if (element_size == 2) {
      reinterpret_cast<uint16_t*>(output)[output_index] =
          reinterpret_cast<const uint16_t*>(table)[table_index];
    } else if (element_size == 1) {
      reinterpret_cast<uint8_t*>(output)[output_index] =
          reinterpret_cast<const uint8_t*>(table)[table_index];
    } else {
      const uint8_t* src =
          reinterpret_cast<const uint8_t*>(table) + table_index * element_size;
      uint8_t* dst =
          reinterpret_cast<uint8_t*>(output) + output_index * element_size;
      for (int32_t byte = 0; byte < element_size; ++byte) {
        dst[byte] = src[byte];
      }
    }
  }
}

}  // namespace

musaError_t LaunchMusaLogBucketizeGatherKernel(
    const void* table, const void* current, const void* sequence, void* output,
    int32_t element_size, int32_t current_element_size,
    int32_t sequence_element_size, int64_t current_count,
    int64_t sequence_count, int64_t table_rows, int64_t block_size,
    float diff_min, float time_scale, float value_min, float log_base,
    float bucket_start, float bucket_span, float bucket_scale,
    int64_t bucket_offset, int64_t bucket_min, int64_t bucket_max,
    float lower_threshold, float upper_threshold, musaStream_t stream) {
  const int64_t output_count = sequence_count * block_size;
  if (output_count == 0) {
    return musaSuccess;
  }
  if (element_size == 2 && block_size == 128) {
    constexpr int kRowsPerBlock = 8;
    constexpr int kThreadsPerOptimizedBlock = 128;
    const int64_t blocks = (sequence_count + kRowsPerBlock - 1) / kRowsPerBlock;
    LogBucketizeGatherElement2Block128Kernel<<<blocks, kThreadsPerOptimizedBlock,
                                               0, stream>>>(
        table, current, sequence, output, current_element_size,
        sequence_element_size, current_count, sequence_count, table_rows,
        diff_min, time_scale, value_min, log_base, bucket_start, bucket_span,
        bucket_scale, bucket_offset, bucket_min, bucket_max, lower_threshold,
        upper_threshold);
    return musaGetLastError();
  }
  LogBucketizeGatherKernel<<<BlocksForCount(output_count), kThreadsPerBlock, 0,
                             stream>>>(
      table, current, sequence, output, element_size, current_element_size,
      sequence_element_size, current_count, sequence_count, table_rows,
      block_size, diff_min, time_scale, value_min, log_base, bucket_start,
      bucket_span, bucket_scale, bucket_offset, bucket_min, bucket_max,
      lower_threshold, upper_threshold, output_count);
  return musaGetLastError();
}
