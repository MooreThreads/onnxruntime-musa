#include "tensor/cast_op_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

constexpr int32_t kOnnxFloat = 1;
constexpr int32_t kOnnxUint8 = 2;
constexpr int32_t kOnnxInt8 = 3;
constexpr int32_t kOnnxUint16 = 4;
constexpr int32_t kOnnxInt16 = 5;
constexpr int32_t kOnnxInt32 = 6;
constexpr int32_t kOnnxInt64 = 7;
constexpr int32_t kOnnxBool = 9;
constexpr int32_t kOnnxFloat16 = 10;
constexpr int32_t kOnnxDouble = 11;
constexpr int32_t kOnnxUint32 = 12;
constexpr int32_t kOnnxUint64 = 13;
constexpr int32_t kOnnxBFloat16 = 16;

__device__ __forceinline__ double CastReadDouble(const void* input,
                                                 int32_t src_type,
                                                 int64_t index) {
  if (src_type == kOnnxFloat16) {
    return static_cast<double>(
        __half2float(reinterpret_cast<const __half*>(input)[index]));
  }
  if (src_type == kOnnxBFloat16) {
    return static_cast<double>(
        __bfloat162float(reinterpret_cast<const __mt_bfloat16*>(input)[index]));
  }
  if (src_type == kOnnxFloat) return reinterpret_cast<const float*>(input)[index];
  if (src_type == kOnnxDouble) return reinterpret_cast<const double*>(input)[index];
  if (src_type == kOnnxUint8) return reinterpret_cast<const uint8_t*>(input)[index];
  if (src_type == kOnnxInt8) return reinterpret_cast<const int8_t*>(input)[index];
  if (src_type == kOnnxUint16) return reinterpret_cast<const uint16_t*>(input)[index];
  if (src_type == kOnnxInt16) return reinterpret_cast<const int16_t*>(input)[index];
  if (src_type == kOnnxInt32) return reinterpret_cast<const int32_t*>(input)[index];
  if (src_type == kOnnxInt64) return static_cast<double>(reinterpret_cast<const int64_t*>(input)[index]);
  if (src_type == kOnnxUint32) return reinterpret_cast<const uint32_t*>(input)[index];
  if (src_type == kOnnxUint64) return static_cast<double>(reinterpret_cast<const uint64_t*>(input)[index]);
  if (src_type == kOnnxBool) return reinterpret_cast<const uint8_t*>(input)[index] ? 1.0 : 0.0;
  return 0.0;
}

__device__ __forceinline__ void CastWriteDouble(void* output, int32_t dst_type,
                                                int64_t index, double value) {
  if (dst_type == kOnnxFloat16) {
    reinterpret_cast<__half*>(output)[index] =
        MusaScalarFromDouble<__half>(value);
  } else if (dst_type == kOnnxBFloat16) {
    reinterpret_cast<__mt_bfloat16*>(output)[index] =
        MusaScalarFromDouble<__mt_bfloat16>(value);
  } else if (dst_type == kOnnxFloat) {
    reinterpret_cast<float*>(output)[index] = static_cast<float>(value);
  } else if (dst_type == kOnnxDouble) {
    reinterpret_cast<double*>(output)[index] = value;
  } else if (dst_type == kOnnxUint8) {
    reinterpret_cast<uint8_t*>(output)[index] = static_cast<uint8_t>(value);
  } else if (dst_type == kOnnxInt8) {
    reinterpret_cast<int8_t*>(output)[index] = static_cast<int8_t>(value);
  } else if (dst_type == kOnnxUint16) {
    reinterpret_cast<uint16_t*>(output)[index] = static_cast<uint16_t>(value);
  } else if (dst_type == kOnnxInt16) {
    reinterpret_cast<int16_t*>(output)[index] = static_cast<int16_t>(value);
  } else if (dst_type == kOnnxInt32) {
    reinterpret_cast<int32_t*>(output)[index] = static_cast<int32_t>(value);
  } else if (dst_type == kOnnxInt64) {
    reinterpret_cast<int64_t*>(output)[index] = static_cast<int64_t>(value);
  } else if (dst_type == kOnnxUint32) {
    reinterpret_cast<uint32_t*>(output)[index] = static_cast<uint32_t>(value);
  } else if (dst_type == kOnnxUint64) {
    reinterpret_cast<uint64_t*>(output)[index] = static_cast<uint64_t>(value);
  } else if (dst_type == kOnnxBool) {
    reinterpret_cast<uint8_t*>(output)[index] =
        static_cast<uint8_t>(value != 0.0);
  }
}

__global__ void CastKernel(const void* input, void* output, int32_t src_type, int32_t dst_type, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    CastWriteDouble(output, dst_type, index,
                    CastReadDouble(input, src_type, index));
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
