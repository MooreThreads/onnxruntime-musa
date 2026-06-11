#include "shared_inc/musa_kernel_common.mu.h"
#include "tensor/split_concat_reorder_impl.h"

namespace {

constexpr int kMaxChunkThreads = 32;

int ClampChunkThreads(int64_t chunks_per_row) {
  if (chunks_per_row < 1) {
    return 1;
  }
  if (chunks_per_row < kMaxChunkThreads) {
    return static_cast<int>(chunks_per_row);
  }
  return kMaxChunkThreads;
}

__global__ void SplitConcatReorderFloat2RowsKernel(
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
      (part * batch + b) * sequence * part_width_chunks + s * part_width_chunks;

  for (int64_t h = threadIdx.x; h < part_width_chunks; h += blockDim.x) {
    output[output_base + h] = input[input_base + h];
  }
}

__global__ void SplitConcatReorderFloatRowsKernel(const float* input,
                                                  float* output, int64_t batch,
                                                  int64_t sequence,
                                                  int64_t part_count,
                                                  int64_t part_width) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  const int64_t s = static_cast<int64_t>(blockIdx.y);
  const int64_t part = static_cast<int64_t>(blockIdx.z);
  if (b >= batch || s >= sequence || part >= part_count) {
    return;
  }

  const int64_t packed_width = part_count * part_width;
  const int64_t input_base =
      (b * sequence + s) * packed_width + part * part_width;
  const int64_t output_base =
      (part * batch + b) * sequence * part_width + s * part_width;

  for (int64_t h = threadIdx.x; h < part_width; h += blockDim.x) {
    output[output_base + h] = input[input_base + h];
  }
}

}  // namespace

musaError_t LaunchMusaSplitConcatReorderFloat(const float* input, float* output,
                                              int64_t batch, int64_t sequence,
                                              int64_t part_count,
                                              int64_t part_width,
                                              musaStream_t stream) {
  const int64_t total_elements = batch * part_count * sequence * part_width;
  if (total_elements == 0) {
    return musaSuccess;
  }

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
    SplitConcatReorderFloat2RowsKernel<<<grid, block, 0, stream>>>(
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
  SplitConcatReorderFloatRowsKernel<<<grid, block, 0, stream>>>(
      input, output, batch, sequence, part_count, part_width);
  return musaGetLastError();
}
