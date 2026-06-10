#include "tensor/concat_split_impl.h"

#include "shared_inc/musa_kernel_common.mu.h"

namespace {

template <typename T>
__global__ void ConcatSplitBatchedCopyKernel(
    const MusaConcatSplitSegment* segments,
    const MusaConcatSplitCopyBlock* blocks, int64_t rows) {
  const MusaConcatSplitCopyBlock block = blocks[blockIdx.x];
  const int64_t segment_idx = block.segment_index;
  const MusaConcatSplitSegment segment = segments[segment_idx];
  const int64_t segment_elements = rows * segment.width;
  const T* input = reinterpret_cast<const T*>(segment.input);
  T* output = reinterpret_cast<T*>(segment.output);

  const int64_t linear = block.element_offset + threadIdx.x;
  if (linear < segment_elements) {
    const int64_t row = linear / segment.width;
    const int64_t col = linear - row * segment.width;
    output[row * segment.output_cols + segment.dst_offset + col] =
        input[row * segment.input_cols + segment.source_offset + col];
  }
}

__global__ void ConcatSplitSumKernel(
    const MusaConcatSplitSumOutput* outputs,
    const MusaConcatSplitSumTerm* terms, int64_t rows, int64_t max_width) {
  const int64_t output_idx = static_cast<int64_t>(blockIdx.y);
  const MusaConcatSplitSumOutput output_spec = outputs[output_idx];

  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       linear < rows * max_width;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = linear / max_width;
    const int64_t col = linear - row * max_width;
    if (col >= output_spec.output_cols) {
      continue;
    }

    float sum = 0.0f;
    for (int64_t term_idx = 0; term_idx < output_spec.term_count; ++term_idx) {
      const MusaConcatSplitSumTerm term =
          terms[output_spec.term_offset + term_idx];
      sum += term.input[row * term.input_cols + term.source_offset + col];
    }
    output_spec.output[row * output_spec.output_cols + col] = sum;
  }
}

template <typename T>
musaError_t LaunchTypedConcatSplitBatchedCopy(
    const MusaConcatSplitSegment* device_segments, int64_t segment_count,
    const MusaConcatSplitCopyBlock* device_blocks, int64_t block_count,
    int64_t rows, musaStream_t stream) {
  if (segment_count == 0 || block_count == 0 || rows == 0) {
    return musaSuccess;
  }
  dim3 grid(static_cast<uint32_t>(block_count), 1, 1);
  ConcatSplitBatchedCopyKernel<T>
      <<<grid, kThreadsPerBlock, 0, stream>>>(device_segments, device_blocks,
                                              rows);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaConcatSplitBatchedCopy(
    const MusaConcatSplitSegment* device_segments, int64_t segment_count,
    const MusaConcatSplitCopyBlock* device_blocks, int64_t block_count,
    int64_t rows, int32_t element_size, musaStream_t stream) {
  switch (element_size) {
    case 1:
      return LaunchTypedConcatSplitBatchedCopy<uint8_t>(
          device_segments, segment_count, device_blocks, block_count, rows,
          stream);
    case 2:
      return LaunchTypedConcatSplitBatchedCopy<uint16_t>(
          device_segments, segment_count, device_blocks, block_count, rows,
          stream);
    case 4:
      return LaunchTypedConcatSplitBatchedCopy<uint32_t>(
          device_segments, segment_count, device_blocks, block_count, rows,
          stream);
    case 8:
      return LaunchTypedConcatSplitBatchedCopy<uint64_t>(
          device_segments, segment_count, device_blocks, block_count, rows,
          stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaConcatSplitSums(
    const MusaConcatSplitSumOutput* device_outputs,
    const MusaConcatSplitSumTerm* device_terms, int64_t output_count,
    int64_t rows, int64_t max_width, musaStream_t stream) {
  if (output_count == 0 || rows == 0 || max_width == 0) {
    return musaSuccess;
  }

  dim3 grid(BlocksForCount(rows * max_width),
            static_cast<uint32_t>(output_count), 1);
  ConcatSplitSumKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
      device_outputs, device_terms, rows, max_width);
  return musaGetLastError();
}
