#include "generator/constant_of_shape_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void ConstantOfShapeKernel(void* output,
                                      uint64_t value,
                                      int32_t element_size,
                                      int64_t count) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    if (element_size == 4) {
      reinterpret_cast<uint32_t*>(output)[index] = static_cast<uint32_t>(value);
    } else if (element_size == 8) {
      reinterpret_cast<uint64_t*>(output)[index] = value;
    } else if (element_size == 1) {
      reinterpret_cast<uint8_t*>(output)[index] = static_cast<uint8_t>(value);
    } else {
      uint8_t* dst = reinterpret_cast<uint8_t*>(output) + index * element_size;
      const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
      for (int32_t byte = 0; byte < element_size; ++byte) {
        dst[byte] = bytes[byte];
      }
    }
  }
}

}  // namespace

musaError_t LaunchMusaConstantOfShapeKernel(void* output,
                                            uint64_t value,
                                            int32_t element_size,
                                            int64_t count,
                                            musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  ConstantOfShapeKernel<<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(
      output, value, element_size, count);
  return musaGetLastError();
}
