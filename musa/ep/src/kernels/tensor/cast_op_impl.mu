#include "tensor/cast_op_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

constexpr int32_t kOnnxFloat = 1;
constexpr int32_t kOnnxInt32 = 6;
constexpr int32_t kOnnxInt64 = 7;
constexpr int32_t kOnnxBool = 9;

__device__ __forceinline__ float CastReadFloat(const void* input, int32_t src_type, int64_t index) {
  if (src_type == kOnnxFloat) return reinterpret_cast<const float*>(input)[index];
  if (src_type == kOnnxInt32) return static_cast<float>(reinterpret_cast<const int32_t*>(input)[index]);
  if (src_type == kOnnxInt64) return static_cast<float>(reinterpret_cast<const int64_t*>(input)[index]);
  if (src_type == kOnnxBool) return reinterpret_cast<const uint8_t*>(input)[index] ? 1.0f : 0.0f;
  return 0.0f;
}

__device__ __forceinline__ int32_t CastReadInt32(const void* input, int32_t src_type, int64_t index) {
  if (src_type == kOnnxFloat) return static_cast<int32_t>(reinterpret_cast<const float*>(input)[index]);
  if (src_type == kOnnxInt32) return reinterpret_cast<const int32_t*>(input)[index];
  if (src_type == kOnnxInt64) return static_cast<int32_t>(reinterpret_cast<const int64_t*>(input)[index]);
  if (src_type == kOnnxBool) return reinterpret_cast<const uint8_t*>(input)[index] ? 1 : 0;
  return 0;
}

__device__ __forceinline__ int64_t CastReadInt64(const void* input, int32_t src_type, int64_t index) {
  if (src_type == kOnnxFloat) return static_cast<int64_t>(reinterpret_cast<const float*>(input)[index]);
  if (src_type == kOnnxInt32) return static_cast<int64_t>(reinterpret_cast<const int32_t*>(input)[index]);
  if (src_type == kOnnxInt64) return reinterpret_cast<const int64_t*>(input)[index];
  if (src_type == kOnnxBool) return reinterpret_cast<const uint8_t*>(input)[index] ? 1 : 0;
  return 0;
}

__device__ __forceinline__ uint8_t CastReadBool(const void* input, int32_t src_type, int64_t index) {
  if (src_type == kOnnxFloat) return static_cast<uint8_t>(reinterpret_cast<const float*>(input)[index] != 0.0f);
  if (src_type == kOnnxInt32) return static_cast<uint8_t>(reinterpret_cast<const int32_t*>(input)[index] != 0);
  if (src_type == kOnnxInt64) return static_cast<uint8_t>(reinterpret_cast<const int64_t*>(input)[index] != 0);
  if (src_type == kOnnxBool) return reinterpret_cast<const uint8_t*>(input)[index];
  return 0;
}

__global__ void CastKernel(const void* input, void* output, int32_t src_type, int32_t dst_type, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    if (dst_type == kOnnxFloat) {
      reinterpret_cast<float*>(output)[index] = CastReadFloat(input, src_type, index);
    } else if (dst_type == kOnnxInt32) {
      reinterpret_cast<int32_t*>(output)[index] = CastReadInt32(input, src_type, index);
    } else if (dst_type == kOnnxInt64) {
      reinterpret_cast<int64_t*>(output)[index] = CastReadInt64(input, src_type, index);
    } else if (dst_type == kOnnxBool) {
      reinterpret_cast<uint8_t*>(output)[index] = CastReadBool(input, src_type, index);
    }
  }
}

__global__ void ExpandKernel(const void* input, void* output, int32_t element_size, MusaBroadcastParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.total_elements; output_index += total_threads) {
    int64_t input_index = 0;
    int64_t unused = 0;
    ResolveBroadcastIndices(output_index, params, input_index, unused);
    if (element_size == 4) {
      reinterpret_cast<uint32_t*>(output)[output_index] = reinterpret_cast<const uint32_t*>(input)[input_index];
    } else if (element_size == 8) {
      reinterpret_cast<uint64_t*>(output)[output_index] = reinterpret_cast<const uint64_t*>(input)[input_index];
    } else if (element_size == 1) {
      reinterpret_cast<uint8_t*>(output)[output_index] = reinterpret_cast<const uint8_t*>(input)[input_index];
    } else {
      const uint8_t* src = reinterpret_cast<const uint8_t*>(input) + input_index * element_size;
      uint8_t* dst = reinterpret_cast<uint8_t*>(output) + output_index * element_size;
      for (int32_t byte = 0; byte < element_size; ++byte) {
        dst[byte] = src[byte];
      }
    }
  }
}

__global__ void CastInt32ToFloatKernel(const int32_t* input,
                                       float* output,
                                       int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = static_cast<float>(input[index]);
  }
}

__global__ void CastInt64ToFloatKernel(const int64_t* input,
                                       float* output,
                                       int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = static_cast<float>(input[index]);
  }
}

}  // namespace

musaError_t LaunchMusaCastKernel(const void* input, void* output, int32_t src_type, int32_t dst_type, int64_t count, musaStream_t stream) {
  if (count == 0) return musaSuccess;
  CastKernel<<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(input, output, src_type, dst_type, count);
  return musaGetLastError();
}

musaError_t LaunchMusaExpandKernel(const void* input, void* output, int32_t element_size, MusaBroadcastParams params, musaStream_t stream) {
  if (params.total_elements == 0) return musaSuccess;
  ExpandKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(input, output, element_size, params);
  return musaGetLastError();
}

musaError_t LaunchMusaCastInt32ToFloatKernel(const int32_t* input,
                                             float* output,
                                             int64_t count,
                                             musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  CastInt32ToFloatKernel<<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(input, output, count);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}

musaError_t LaunchMusaCastInt64ToFloatKernel(const int64_t* input,
                                             float* output,
                                             int64_t count,
                                             musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  CastInt64ToFloatKernel<<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(input, output, count);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}
