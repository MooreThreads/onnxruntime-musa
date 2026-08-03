#include "reduction/rec_rank_calibration_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void RecRankCalibrationKernel(
    const double* task_relu, const double* task_scores, float* output,
    int64_t rows, int64_t bucket_size, float epsilon,
    bool complement_from_clipped_probability) {
  __shared__ double partial[kThreadsPerBlock];

  for (int64_t row = static_cast<int64_t>(blockIdx.x); row < rows;
       row += gridDim.x) {
    const double scaled = task_scores[row] * static_cast<double>(bucket_size);
    const double left = floor(scaled);
    const double frac = scaled - left;
    const double* row_relu = task_relu + row * bucket_size;

    double sum = 0.0;
    for (int64_t bucket = threadIdx.x; bucket < bucket_size;
         bucket += blockDim.x) {
      const double bucket_value = static_cast<double>(bucket);
      // Keep the explicit multiply used by the ONNX graph. In particular,
      // 0 * NaN must remain NaN instead of being hidden by a conditional.
      const double weight = static_cast<double>(bucket_value < left) +
                            static_cast<double>(bucket_value == left) * frac;
      sum += row_relu[bucket] * weight / static_cast<double>(bucket_size);
    }

    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = kThreadsPerBlock / 2; offset > 0; offset >>= 1) {
      if (threadIdx.x < offset) {
        partial[threadIdx.x] += partial[threadIdx.x + offset];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      // The source graph casts the double reduction to float before Clip.
      // Comparisons (rather than fmin/fmax) preserve NaN like ONNX Clip.
      const float raw_probability = static_cast<float>(partial[0]);
      float probability = raw_probability;
      if (probability < epsilon) {
        probability = epsilon;
      }
      if (probability > 1.0f) {
        probability = 1.0f;
      }
      const float complement_probability =
          complement_from_clipped_probability ? probability : raw_probability;
      float complement = 1.0f - complement_probability;
      if (complement < epsilon) {
        complement = epsilon;
      }
      if (complement > 1.0f) {
        complement = 1.0f;
      }
      output[row] = logf(probability / complement);
    }
    __syncthreads();
  }
}

}  // namespace

musaError_t LaunchMusaRecRankCalibrationKernel(
    const double* task_relu, const double* task_scores, float* output,
    int64_t rows, int64_t bucket_size, float epsilon,
    bool complement_from_clipped_probability, musaStream_t stream) {
  if (rows == 0) {
    return musaSuccess;
  }
  if (bucket_size <= 0) {
    return musaErrorInvalidValue;
  }
  const int blocks = static_cast<int>(rows < kMaxBlocks ? rows : kMaxBlocks);
  RecRankCalibrationKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      task_relu, task_scores, output, rows, bucket_size, epsilon,
      complement_from_clipped_probability);
  return musaGetLastError();
}
