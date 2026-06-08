#include "tensor/split_impl.h"

namespace {

constexpr size_t ToBytes(int64_t elements, int32_t element_size) {
  return static_cast<size_t>(elements) * static_cast<size_t>(element_size);
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
