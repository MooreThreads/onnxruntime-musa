#include "tensor/slice_sum_concat_impl.h"

#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void SliceSumConcatFloatKernel(MusaSliceSumConcatParams params,
                                          float* output) {
  const int64_t total = params.batch * params.output_cols;
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t linear = thread_id; linear < total; linear += total_threads) {
    const int64_t row = linear / params.output_cols;
    const int64_t col = linear - row * params.output_cols;

    float value = 0.0f;
    for (int32_t seg_idx = 0; seg_idx < params.segment_count; ++seg_idx) {
      const MusaSliceSumConcatSegment& segment = params.segments[seg_idx];
      if (col < segment.dst_col || col >= segment.dst_col + segment.width) {
        continue;
      }

      const int64_t local_col = col - segment.dst_col;
      if (segment.mode ==
          static_cast<int32_t>(MusaSliceSumConcatSegmentMode::Direct)) {
        value = segment.direct_input[row * segment.direct_input_cols + local_col];
      } else {
        float acc = 0.0f;
        for (int32_t i = 0; i < segment.slice_count; ++i) {
          const MusaSliceSumConcatSlice& slice =
              params.slices[segment.slice_start + i];
          acc += slice.input[row * slice.input_cols + slice.start_col + local_col];
        }
        value = acc;
      }
      break;
    }
    output[linear] = value;
  }
}

}  // namespace

musaError_t LaunchMusaSliceSumConcatFloat(
    const MusaSliceSumConcatParams& params, float* output, musaStream_t stream) {
  if (params.batch == 0 || params.output_cols == 0) {
    return musaSuccess;
  }
  SliceSumConcatFloatKernel<<<BlocksForCount(params.batch * params.output_cols),
                              kThreadsPerBlock, 0, stream>>>(params, output);
  return musaGetLastError();
}
