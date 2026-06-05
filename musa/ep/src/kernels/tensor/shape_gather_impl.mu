#include "tensor/shape_gather_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

constexpr int32_t kOnnxInt32 = 6;
constexpr int32_t kOnnxInt64 = 7;

__global__ void ShapeGatherKernel(void* output, MusaShapeGatherParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = thread_id; i < params.output_count; i += total_threads) {
    int64_t index = params.indices[i];
    if (index < 0) {
      index += params.rank;
    }
    int64_t value = 0;
    if (index >= 0 && index < params.rank) {
      value = params.dims[index];
    }
    if (params.output_type == kOnnxInt32) {
      reinterpret_cast<int32_t*>(output)[i] = static_cast<int32_t>(value);
    } else if (params.output_type == kOnnxInt64) {
      reinterpret_cast<int64_t*>(output)[i] = value;
    }
  }
}

}  // namespace

musaError_t LaunchMusaShapeGatherKernel(void* output,
                                        MusaShapeGatherParams params,
                                        musaStream_t stream) {
  if (params.output_count == 0) {
    return musaSuccess;
  }
  ShapeGatherKernel<<<BlocksForCount(params.output_count), kThreadsPerBlock, 0,
                      stream>>>(output, params);
  return musaGetLastError();
}
