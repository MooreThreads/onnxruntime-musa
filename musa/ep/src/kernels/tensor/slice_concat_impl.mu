#include "tensor/slice_concat_impl.h"

#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void SliceConcatKernel(const MusaSliceConcatSegment* segments,
                                  int64_t segment_count, float* output,
                                  int64_t rows, int64_t output_cols) {
  const int64_t total_elements = rows * output_cols;
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t linear = thread_id; linear < total_elements;
       linear += total_threads) {
    const int64_t row = linear / output_cols;
    const int64_t col = linear - row * output_cols;

    int64_t lo = 0;
    int64_t hi = segment_count;
    while (lo < hi) {
      const int64_t mid = lo + (hi - lo) / 2;
      const MusaSliceConcatSegment segment = segments[mid];
      if (col < segment.dst_offset + segment.width) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }

    const MusaSliceConcatSegment segment = segments[lo];
    const int64_t local_col = col - segment.dst_offset;
    if (segment.zero_fill != 0) {
      output[linear] = 0.0f;
      continue;
    }
    output[linear] =
        segment.input[row * segment.input_cols + segment.start_col + local_col];
  }
}

__global__ void SliceConcatEqualWidthKernel(
    const MusaSliceConcatSegment* segments, int64_t segment_count,
    float* output, int64_t rows, int64_t output_cols, int64_t segment_width,
    int32_t width_shift) {
  const int64_t segment_index = static_cast<int64_t>(blockIdx.y);
  if (segment_index >= segment_count) {
    return;
  }
  const MusaSliceConcatSegment segment = segments[segment_index];
  const int64_t total_elements = rows * segment_width;
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t local_col_mask = segment_width - 1;

  for (int64_t linear = thread_id; linear < total_elements;
       linear += total_threads) {
    const int64_t row = linear >> width_shift;
    const int64_t local_col = linear & local_col_mask;
    const int64_t output_index =
        row * output_cols + segment.dst_offset + local_col;
    if (segment.zero_fill != 0) {
      output[output_index] = 0.0f;
      continue;
    }
    output[output_index] =
        segment.input[row * segment.input_cols + segment.start_col + local_col];
  }
}

__global__ void SliceConcatSegmentedKernel(
    const MusaSliceConcatSegment* segments, int64_t segment_count,
    float* output, int64_t rows, int64_t output_cols) {
  const int64_t block = static_cast<int64_t>(blockIdx.x);

  int64_t lo = 0;
  int64_t hi = segment_count;
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo) / 2;
    const MusaSliceConcatSegment segment = segments[mid];
    if (block < segment.block_offset + segment.block_count) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  if (lo >= segment_count) {
    return;
  }

  const MusaSliceConcatSegment segment = segments[lo];
  const int64_t local_block = block - segment.block_offset;
  int64_t linear = local_block * blockDim.x + threadIdx.x;
  const int64_t total_elements = rows * segment.width;
  if (linear >= total_elements) {
    return;
  }

  const int64_t row = linear / segment.width;
  const int64_t local_col = linear - row * segment.width;
  const int64_t output_index =
      row * output_cols + segment.dst_offset + local_col;
  if (segment.zero_fill != 0) {
    output[output_index] = 0.0f;
    return;
  }
  output[output_index] =
      segment.input[row * segment.input_cols + segment.start_col + local_col];
}

}  // namespace

musaError_t LaunchMusaSliceConcatKernel(
    const MusaSliceConcatSegment* device_segments, int64_t segment_count,
    float* output, int64_t rows, int64_t output_cols, musaStream_t stream) {
  const int64_t total_elements = rows * output_cols;
  if (total_elements == 0) {
    return musaSuccess;
  }
  SliceConcatKernel<<<BlocksForCount(total_elements), kThreadsPerBlock, 0,
                      stream>>>(device_segments, segment_count, output, rows,
                                output_cols);
  return musaGetLastError();
}

musaError_t LaunchMusaSliceConcatSegmentedKernel(
    const MusaSliceConcatSegment* device_segments, int64_t segment_count,
    int64_t total_blocks, float* output, int64_t rows, int64_t output_cols,
    musaStream_t stream) {
  if (rows * output_cols == 0 || total_blocks == 0) {
    return musaSuccess;
  }
  SliceConcatSegmentedKernel<<<static_cast<uint32_t>(total_blocks),
                               kSliceConcatThreadsPerBlock, 0, stream>>>(
      device_segments, segment_count, output, rows, output_cols);
  return musaGetLastError();
}

musaError_t LaunchMusaSliceConcatEqualWidthKernel(
    const MusaSliceConcatSegment* device_segments, int64_t segment_count,
    float* output, int64_t rows, int64_t output_cols, int64_t segment_width,
    int32_t width_shift, musaStream_t stream) {
  const int64_t total_elements = rows * output_cols;
  if (total_elements == 0) {
    return musaSuccess;
  }
  dim3 grid(BlocksForCount(rows * segment_width),
            static_cast<uint32_t>(segment_count), 1);
  SliceConcatEqualWidthKernel<<<grid, kSliceConcatThreadsPerBlock, 0, stream>>>(
      device_segments, segment_count, output, rows, output_cols, segment_width,
      width_shift);
  return musaGetLastError();
}
