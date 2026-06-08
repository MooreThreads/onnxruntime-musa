#include "tensor/split_impl.h"

#include <cstdint>

namespace {

constexpr int64_t kBatchedSplitOutputThreshold = 32;
constexpr int kThreadsPerBlock = 256;

constexpr size_t ToBytes(int64_t elements, int32_t element_size) {
  return static_cast<size_t>(elements) * static_cast<size_t>(element_size);
}

template <typename T>
__global__ void SplitBatchedCopyKernel(const T* input, T** outputs,
                                       const int64_t* split_sizes,
                                       const int64_t* split_offsets,
                                       int64_t outer, int64_t inner,
                                       int64_t input_axis) {
  const int64_t output_idx = static_cast<int64_t>(blockIdx.y);
  const int64_t split_size = split_sizes[output_idx];
  const int64_t output_elements = outer * split_size * inner;
  T* output = outputs[output_idx];
  const int64_t axis_offset = split_offsets[output_idx];

  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       linear < output_elements;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t outer_idx = linear / (split_size * inner);
    const int64_t local = linear - outer_idx * split_size * inner;
    const int64_t input_idx =
        (outer_idx * input_axis + axis_offset) * inner + local;
    output[linear] = input[input_idx];
  }
}

template <typename T>
musaError_t LaunchTypedSplitBatchedCopy(const void* input, void* const* outputs,
                                        const int64_t* split_sizes,
                                        const int64_t* split_offsets,
                                        int64_t output_count, int64_t outer,
                                        int64_t inner, int64_t input_axis,
                                        musaStream_t stream) {
  T** device_outputs = nullptr;
  int64_t* device_split_sizes = nullptr;
  int64_t* device_split_offsets = nullptr;

  const size_t outputs_bytes =
      static_cast<size_t>(output_count) * sizeof(void*);
  const size_t split_bytes = static_cast<size_t>(output_count) * sizeof(int64_t);

  musaError_t status =
      musaMalloc(reinterpret_cast<void**>(&device_outputs), outputs_bytes);
  if (status != musaSuccess) return status;
  status =
      musaMalloc(reinterpret_cast<void**>(&device_split_sizes), split_bytes);
  if (status != musaSuccess) {
    (void)musaFree(device_outputs);
    return status;
  }
  status =
      musaMalloc(reinterpret_cast<void**>(&device_split_offsets), split_bytes);
  if (status != musaSuccess) {
    (void)musaFree(device_split_sizes);
    (void)musaFree(device_outputs);
    return status;
  }

  status = musaMemcpyAsync(device_outputs, outputs, outputs_bytes,
                           musaMemcpyHostToDevice, stream);
  if (status == musaSuccess) {
    status = musaMemcpyAsync(device_split_sizes, split_sizes, split_bytes,
                             musaMemcpyHostToDevice, stream);
  }
  if (status == musaSuccess) {
    status = musaMemcpyAsync(device_split_offsets, split_offsets, split_bytes,
                             musaMemcpyHostToDevice, stream);
  }
  if (status == musaSuccess) {
    int64_t max_output_elements = 0;
    for (int64_t i = 0; i < output_count; ++i) {
      const int64_t output_elements = outer * split_sizes[i] * inner;
      if (output_elements > max_output_elements) {
        max_output_elements = output_elements;
      }
    }
    int64_t grid_x_i64 =
        (max_output_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (grid_x_i64 < 1) {
      grid_x_i64 = 1;
    }
    const uint32_t grid_x = static_cast<uint32_t>(grid_x_i64);
    dim3 grid(grid_x, static_cast<uint32_t>(output_count), 1);
    SplitBatchedCopyKernel<T><<<grid, kThreadsPerBlock, 0, stream>>>(
        static_cast<const T*>(input), device_outputs, device_split_sizes,
        device_split_offsets, outer, inner, input_axis);
    status = musaGetLastError();
  }

  musaError_t free_offsets_status = musaFree(device_split_offsets);
  musaError_t free_sizes_status = musaFree(device_split_sizes);
  musaError_t free_outputs_status = musaFree(device_outputs);
  if (status != musaSuccess) return status;
  if (free_offsets_status != musaSuccess) return free_offsets_status;
  if (free_sizes_status != musaSuccess) return free_sizes_status;
  return free_outputs_status;
}

}  // namespace

musaError_t LaunchMusaSplitCopies(const void* input, void* const* outputs,
                                  const int64_t* split_sizes,
                                  int64_t output_count, int64_t outer,
                                  int64_t inner, int64_t input_axis,
                                  int32_t element_size,
                                  musaStream_t stream) {
  const auto* src_base = static_cast<const uint8_t*>(input);
  int64_t axis_start = 0;
  const size_t src_pitch = ToBytes(input_axis * inner, element_size);
  for (int64_t output_idx = 0; output_idx < output_count; ++output_idx) {
    const int64_t split_size = split_sizes[output_idx];
    const size_t width_bytes = ToBytes(split_size * inner, element_size);
    if (width_bytes == 0 || outer == 0) {
      continue;
    }
    const auto* src = src_base + ToBytes(axis_start * inner, element_size);
    musaError_t status = musaMemcpy2DAsync(
        outputs[output_idx], width_bytes, src, src_pitch, width_bytes,
        static_cast<size_t>(outer), musaMemcpyDeviceToDevice, stream);
    if (status != musaSuccess) {
      return status;
    }
    axis_start += split_size;
  }
  return musaSuccess;
}

musaError_t LaunchMusaSplitBatchedCopy(const void* input, void* const* outputs,
                                       const int64_t* split_sizes,
                                       const int64_t* split_offsets,
                                       int64_t output_count, int64_t outer,
                                       int64_t inner, int64_t input_axis,
                                       int32_t element_size,
                                       musaStream_t stream) {
  if (output_count < kBatchedSplitOutputThreshold) {
    return LaunchMusaSplitCopies(input, outputs, split_sizes, output_count,
                                 outer, inner, input_axis, element_size,
                                 stream);
  }

  switch (element_size) {
    case 1:
      return LaunchTypedSplitBatchedCopy<uint8_t>(
          input, outputs, split_sizes, split_offsets, output_count, outer,
          inner, input_axis, stream);
    case 2:
      return LaunchTypedSplitBatchedCopy<uint16_t>(
          input, outputs, split_sizes, split_offsets, output_count, outer,
          inner, input_axis, stream);
    case 4:
      return LaunchTypedSplitBatchedCopy<uint32_t>(
          input, outputs, split_sizes, split_offsets, output_count, outer,
          inner, input_axis, stream);
    case 8:
      return LaunchTypedSplitBatchedCopy<uint64_t>(
          input, outputs, split_sizes, split_offsets, output_count, outer,
          inner, input_axis, stream);
    default:
      return LaunchMusaSplitCopies(input, outputs, split_sizes, output_count,
                                   outer, inner, input_axis, element_size,
                                   stream);
  }
}
