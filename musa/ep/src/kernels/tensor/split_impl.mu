#include "tensor/split_impl.h"

namespace {

constexpr size_t ToBytes(int64_t elements, int32_t element_size) {
  return static_cast<size_t>(elements) * static_cast<size_t>(element_size);
}

constexpr int kSplitCopyThreads = 256;

__global__ void SplitManySmallCopyKernel(
    const uint8_t* input, const MusaSplitCopyDesc* descriptors,
    int64_t output_count, int64_t outer, int64_t inner, int64_t input_axis,
    int32_t element_size) {
  const int64_t output_idx = static_cast<int64_t>(blockIdx.x);
  if (output_idx >= output_count) {
    return;
  }

  const MusaSplitCopyDesc desc = descriptors[output_idx];
  const int64_t row_elements = desc.split_size * inner;
  if (row_elements <= 0) {
    return;
  }
  const int64_t total_elements = outer * row_elements;
  auto* output = static_cast<uint8_t*>(desc.output);

  for (int64_t output_element = threadIdx.x; output_element < total_elements;
       output_element += blockDim.x) {
    const int64_t outer_idx = output_element / row_elements;
    const int64_t row_offset = output_element - outer_idx * row_elements;
    const int64_t input_element =
        (outer_idx * input_axis + desc.axis_start) * inner + row_offset;

    if (element_size == 4) {
      reinterpret_cast<uint32_t*>(output)[output_element] =
          reinterpret_cast<const uint32_t*>(input)[input_element];
    } else if (element_size == 8) {
      reinterpret_cast<uint64_t*>(output)[output_element] =
          reinterpret_cast<const uint64_t*>(input)[input_element];
    } else if (element_size == 2) {
      reinterpret_cast<uint16_t*>(output)[output_element] =
          reinterpret_cast<const uint16_t*>(input)[input_element];
    } else if (element_size == 1) {
      output[output_element] = input[input_element];
    } else {
      const auto* src = input + input_element * element_size;
      auto* dst = output + output_element * element_size;
      for (int32_t byte = 0; byte < element_size; ++byte) {
        dst[byte] = src[byte];
      }
    }
  }
}

__global__ void SplitManySmallRowsKernel(
    const uint8_t* input, const MusaSplitElementDesc* descriptors,
    int64_t outer, int64_t input_row_elements, int32_t element_size) {
  const int64_t outer_idx = static_cast<int64_t>(blockIdx.y);
  const int64_t row_element =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (outer_idx >= outer || row_element >= input_row_elements) {
    return;
  }

  const MusaSplitElementDesc desc = descriptors[row_element];
  auto* output = static_cast<uint8_t*>(desc.output);
  const int64_t input_element = outer_idx * input_row_elements + row_element;
  const int64_t output_element =
      outer_idx * desc.output_width + desc.local_element;

  if (element_size == 4) {
    reinterpret_cast<uint32_t*>(output)[output_element] =
        reinterpret_cast<const uint32_t*>(input)[input_element];
  } else if (element_size == 8) {
    reinterpret_cast<uint64_t*>(output)[output_element] =
        reinterpret_cast<const uint64_t*>(input)[input_element];
  } else if (element_size == 2) {
    reinterpret_cast<uint16_t*>(output)[output_element] =
        reinterpret_cast<const uint16_t*>(input)[input_element];
  } else if (element_size == 1) {
    output[output_element] = input[input_element];
  } else {
    const auto* src = input + input_element * element_size;
    auto* dst = output + output_element * element_size;
    for (int32_t byte = 0; byte < element_size; ++byte) {
      dst[byte] = src[byte];
    }
  }
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


musaError_t LaunchMusaSplitManySmallCopies(
    const void* input, const MusaSplitCopyDesc* descriptors,
    int64_t output_count, int64_t outer, int64_t inner, int64_t input_axis,
    int32_t element_size, musaStream_t stream) {
  if (output_count == 0 || outer == 0) {
    return musaSuccess;
  }
  SplitManySmallCopyKernel<<<static_cast<unsigned int>(output_count),
                             kSplitCopyThreads, 0, stream>>>(
      static_cast<const uint8_t*>(input), descriptors, output_count, outer,
      inner, input_axis, element_size);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}


musaError_t LaunchMusaSplitManySmallRows(
    const void* input, const MusaSplitElementDesc* descriptors,
    int64_t outer, int64_t input_row_elements, int32_t element_size,
    musaStream_t stream) {
  if (outer == 0 || input_row_elements == 0) {
    return musaSuccess;
  }
  const auto blocks_per_row = static_cast<unsigned int>(
      (input_row_elements + kSplitCopyThreads - 1) / kSplitCopyThreads);
  dim3 grid(blocks_per_row, static_cast<unsigned int>(outer));
  SplitManySmallRowsKernel<<<grid, kSplitCopyThreads, 0, stream>>>(
      static_cast<const uint8_t*>(input), descriptors, outer,
      input_row_elements, element_size);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}
