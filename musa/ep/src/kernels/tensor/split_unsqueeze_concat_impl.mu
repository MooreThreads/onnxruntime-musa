#include "shared_inc/musa_kernel_common.mu.h"
#include "tensor/split_unsqueeze_concat_impl.h"

namespace {

constexpr int kMaxChunkThreads = 32;
constexpr int kTileDim = 16;

int ClampChunkThreads(int64_t chunks_per_row) {
  if (chunks_per_row < 1) {
    return 1;
  }
  if (chunks_per_row < kMaxChunkThreads) {
    return static_cast<int>(chunks_per_row);
  }
  return kMaxChunkThreads;
}

__global__ void SplitUnsqueezeConcatFloat2RowsKernel(
    const uint64_t* input, uint64_t* output, int64_t batch, int64_t sequence,
    int64_t part_count, int64_t part_width_chunks) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  const int64_t s = static_cast<int64_t>(blockIdx.y);
  const int64_t part = static_cast<int64_t>(blockIdx.z);
  if (b >= batch || s >= sequence || part >= part_count) {
    return;
  }

  const int64_t packed_width_chunks = part_count * part_width_chunks;
  const int64_t input_base =
      (b * sequence + s) * packed_width_chunks + part * part_width_chunks;
  const int64_t output_base =
      ((part * batch + b) * sequence + s) * part_width_chunks;

  for (int64_t h = threadIdx.x; h < part_width_chunks; h += blockDim.x) {
    output[output_base + h] = input[input_base + h];
  }
}

__global__ void SplitUnsqueezeConcatFloatRowsKernel(
    const float* input, float* output, int64_t batch, int64_t sequence,
    int64_t part_count, int64_t part_width) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  const int64_t s = static_cast<int64_t>(blockIdx.y);
  const int64_t part = static_cast<int64_t>(blockIdx.z);
  if (b >= batch || s >= sequence || part >= part_count) {
    return;
  }

  const int64_t packed_width = part_count * part_width;
  const int64_t input_base =
      (b * sequence + s) * packed_width + part * part_width;
  const int64_t output_base = ((part * batch + b) * sequence + s) * part_width;

  for (int64_t h = threadIdx.x; h < part_width; h += blockDim.x) {
    output[output_base + h] = input[input_base + h];
  }
}

__global__ void SplitUnsqueezeConcatTransposeTiledKernel(
    const float* input, float* output, int64_t batch, int64_t sequence,
    int64_t part_count, int64_t part_width) {
  __shared__ float tile[kTileDim][kTileDim + 1];

  const int64_t part_batch = static_cast<int64_t>(blockIdx.z);
  const int64_t part = part_batch / batch;
  const int64_t b = part_batch - part * batch;
  const int64_t input_s =
      static_cast<int64_t>(blockIdx.y) * kTileDim + threadIdx.y;
  const int64_t input_h =
      static_cast<int64_t>(blockIdx.x) * kTileDim + threadIdx.x;

  if (part < part_count && b < batch && input_s < sequence &&
      input_h < part_width) {
    const int64_t packed_width = part_count * part_width;
    const int64_t input_index =
        (b * sequence + input_s) * packed_width + part * part_width + input_h;
    tile[threadIdx.y][threadIdx.x] = input[input_index];
  }
  __syncthreads();

  const int64_t output_h =
      static_cast<int64_t>(blockIdx.x) * kTileDim + threadIdx.y;
  const int64_t output_s =
      static_cast<int64_t>(blockIdx.y) * kTileDim + threadIdx.x;
  if (part < part_count && b < batch && output_h < part_width &&
      output_s < sequence) {
    const int64_t output_index =
        ((part * batch + b) * part_width + output_h) * sequence + output_s;
    output[output_index] = tile[threadIdx.x][threadIdx.y];
  }
}

musaError_t LaunchSplitUnsqueezeConcatNoTranspose(
    const float* input, float* output, int64_t batch, int64_t sequence,
    int64_t part_count, int64_t part_width, musaStream_t stream) {
  if ((part_width & 1) == 0) {
    const int64_t part_width_chunks = part_width >> 1;
    const int chunk_threads = ClampChunkThreads(part_width_chunks);
    const int rows_per_block = kThreadsPerBlock / chunk_threads;
    const dim3 block(static_cast<unsigned int>(chunk_threads),
                     static_cast<unsigned int>(rows_per_block), 1);
    const dim3 grid(static_cast<unsigned int>((batch + rows_per_block - 1) /
                                              rows_per_block),
                    static_cast<unsigned int>(sequence),
                    static_cast<unsigned int>(part_count));
    SplitUnsqueezeConcatFloat2RowsKernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(input),
        reinterpret_cast<uint64_t*>(output), batch, sequence, part_count,
        part_width_chunks);
    return musaGetLastError();
  }

  const int chunk_threads = ClampChunkThreads(part_width);
  const int rows_per_block = kThreadsPerBlock / chunk_threads;
  const dim3 block(static_cast<unsigned int>(chunk_threads),
                   static_cast<unsigned int>(rows_per_block), 1);
  const dim3 grid(
      static_cast<unsigned int>((batch + rows_per_block - 1) / rows_per_block),
      static_cast<unsigned int>(sequence),
      static_cast<unsigned int>(part_count));
  SplitUnsqueezeConcatFloatRowsKernel<<<grid, block, 0, stream>>>(
      input, output, batch, sequence, part_count, part_width);
  return musaGetLastError();
}

musaError_t LaunchSplitUnsqueezeConcatTranspose(
    const float* input, float* output, int64_t batch, int64_t sequence,
    int64_t part_count, int64_t part_width, musaStream_t stream) {
  const dim3 block(kTileDim, kTileDim, 1);
  const dim3 grid(
      static_cast<unsigned int>((part_width + kTileDim - 1) / kTileDim),
      static_cast<unsigned int>((sequence + kTileDim - 1) / kTileDim),
      static_cast<unsigned int>(batch * part_count));
  SplitUnsqueezeConcatTransposeTiledKernel<<<grid, block, 0, stream>>>(
      input, output, batch, sequence, part_count, part_width);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaSplitUnsqueezeConcatFloat(
    const float* input, float* output, int64_t batch, int64_t sequence,
    int64_t part_count, int64_t part_width, bool transpose,
    musaStream_t stream) {
  const int64_t total_elements = batch * part_count * sequence * part_width;
  if (total_elements == 0) {
    return musaSuccess;
  }

  if (transpose) {
    return LaunchSplitUnsqueezeConcatTranspose(input, output, batch, sequence,
                                               part_count, part_width, stream);
  }
  return LaunchSplitUnsqueezeConcatNoTranspose(input, output, batch, sequence,
                                               part_count, part_width, stream);
}
