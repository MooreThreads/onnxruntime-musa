#include "tensor/onehot_impl.h"

#include "shared_inc/musa_kernel_common.mu.h"

namespace {

constexpr int32_t kOnnxTensorElementDataTypeInt32 = 6;
constexpr int32_t kOnnxTensorElementDataTypeInt64 = 7;

template <typename IndexT>
__global__ void OneHotKernel(const IndexT* indices,
                             void* output,
                             int32_t element_size,
                             uint64_t off_value,
                             uint64_t on_value,
                             int64_t depth,
                             int64_t suffix,
                             int64_t total_elements) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t depth_suffix = depth * suffix;
  for (int64_t id = thread_id; id < total_elements; id += total_threads) {
    const int64_t prefix = id / depth_suffix;
    const int64_t prefix_offset = id - prefix * depth_suffix;
    const int64_t depth_index = prefix_offset / suffix;
    const int64_t suffix_index = prefix_offset - depth_index * suffix;
    const int64_t indices_index = prefix * suffix + suffix_index;
    const int64_t raw_index = static_cast<int64_t>(indices[indices_index]);
    const bool valid = raw_index >= -depth && raw_index < depth;
    const int64_t adjusted_index = raw_index >= 0 ? raw_index : raw_index + depth;
    const uint64_t value =
        valid && adjusted_index == depth_index ? on_value : off_value;

    if (element_size == 8) {
      reinterpret_cast<uint64_t*>(output)[id] = value;
    } else if (element_size == 4) {
      reinterpret_cast<uint32_t*>(output)[id] = static_cast<uint32_t>(value);
    } else if (element_size == 2) {
      reinterpret_cast<uint16_t*>(output)[id] = static_cast<uint16_t>(value);
    } else {
      reinterpret_cast<uint8_t*>(output)[id] = static_cast<uint8_t>(value);
    }
  }
}

template <typename IndexT>
musaError_t LaunchTypedOneHotKernel(const void* indices,
                                    void* output,
                                    int32_t element_size,
                                    uint64_t off_value,
                                    uint64_t on_value,
                                    int64_t depth,
                                    int64_t suffix,
                                    int64_t total_elements,
                                    musaStream_t stream) {
  if (total_elements == 0) {
    return musaSuccess;
  }
  OneHotKernel<IndexT><<<BlocksForCount(total_elements), kThreadsPerBlock, 0,
                         stream>>>(
      reinterpret_cast<const IndexT*>(indices), output, element_size, off_value,
      on_value, depth, suffix, total_elements);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaOneHotKernel(const void* indices,
                                   void* output,
                                   int32_t indices_type,
                                   int32_t element_size,
                                   uint64_t off_value,
                                   uint64_t on_value,
                                   int64_t depth,
                                   int64_t suffix,
                                   int64_t total_elements,
                                   musaStream_t stream) {
  if (indices_type == kOnnxTensorElementDataTypeInt64) {
    return LaunchTypedOneHotKernel<int64_t>(
        indices, output, element_size, off_value, on_value, depth, suffix,
        total_elements, stream);
  }
  if (indices_type == kOnnxTensorElementDataTypeInt32) {
    return LaunchTypedOneHotKernel<int32_t>(
        indices, output, element_size, off_value, on_value, depth, suffix,
        total_elements, stream);
  }
  return musaErrorInvalidValue;
}
