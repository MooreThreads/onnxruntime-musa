#include "tensor/concat_impl.h"

namespace {

constexpr size_t ToBytes(int64_t elements, int32_t element_size) {
  return static_cast<size_t>(elements) * static_cast<size_t>(element_size);
}

}  // namespace

musaError_t LaunchMusaConcatCopies(void* output, const void* const* inputs,
                                   const int64_t* input_axis_dims,
                                   int64_t input_count, int64_t outer,
                                   int64_t inner, int64_t output_axis,
                                   int32_t element_size,
                                   musaStream_t stream) {
  auto* dst_base = static_cast<uint8_t*>(output);
  int64_t dst_axis_offset = 0;
  const size_t dst_pitch = ToBytes(output_axis * inner, element_size);
  for (int64_t input_idx = 0; input_idx < input_count; ++input_idx) {
    const int64_t input_axis = input_axis_dims[input_idx];
    const size_t width_bytes = ToBytes(input_axis * inner, element_size);
    if (width_bytes == 0 || outer == 0) {
      continue;
    }
    auto* dst = dst_base + ToBytes(dst_axis_offset * inner, element_size);
    musaError_t status = musaMemcpy2DAsync(
        dst, dst_pitch, inputs[input_idx], width_bytes, width_bytes,
        static_cast<size_t>(outer), musaMemcpyDeviceToDevice, stream);
    if (status != musaSuccess) {
      return status;
    }
    dst_axis_offset += input_axis;
  }
  return musaSuccess;
}
