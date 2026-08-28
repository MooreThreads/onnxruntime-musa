#include "tensor/concat_impl.h"

namespace {

constexpr size_t ToBytes(int64_t elements, int32_t element_size) {
  return static_cast<size_t>(elements) * static_cast<size_t>(element_size);
}

constexpr int kConcatCopyThreads = 256;

struct MusaConcatBf16HighArityParams {
  const uint4* inputs[kMusaConcatBf16HighArityMaxInputs];
  int32_t input_row_vectors[kMusaConcatBf16HighArityMaxInputs];
  int32_t output_offsets[kMusaConcatBf16HighArityMaxInputs];
  int64_t outer;
  int32_t output_row_vectors;
};

__global__ void ConcatBf16LastAxisHighArityKernel(
    uint4* output, MusaConcatBf16HighArityParams params) {
  const int64_t input_idx = static_cast<int64_t>(blockIdx.y);
  const int64_t input_row_vectors = params.input_row_vectors[input_idx];
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t input_vectors = params.outer * input_row_vectors;
  if (index >= input_vectors) {
    return;
  }

  const int64_t row = index / input_row_vectors;
  const int64_t column = index - row * input_row_vectors;
  output[row * params.output_row_vectors + params.output_offsets[input_idx] +
         column] = params.inputs[input_idx][index];
}

struct MusaConcatSmallRowsParams {
  const void* inputs[kMusaConcatSmallRowsMaxInputs];
  int64_t input_axis_dims[kMusaConcatSmallRowsMaxInputs];
  int64_t input_count;
  int64_t inner;
};

__device__ void CopyConcatElement(uint8_t* output, int64_t output_element,
                                  const uint8_t* input, int64_t input_element,
                                  int32_t element_size) {
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

__global__ void ConcatManySmallRowsDirectKernel(
    uint8_t* output, MusaConcatSmallRowsParams params, int64_t outer,
    int64_t output_row_elements, int32_t element_size) {
  const int64_t outer_idx = static_cast<int64_t>(blockIdx.y);
  const int64_t row_element =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (outer_idx >= outer || row_element >= output_row_elements) {
    return;
  }

  int64_t input_offset = 0;
  for (int64_t input_idx = 0; input_idx < params.input_count; ++input_idx) {
    const int64_t input_width =
        params.input_axis_dims[input_idx] * params.inner;
    const int64_t input_end = input_offset + input_width;
    if (row_element < input_end) {
      const auto* input = static_cast<const uint8_t*>(params.inputs[input_idx]);
      const int64_t local_element = row_element - input_offset;
      const int64_t input_element = outer_idx * input_width + local_element;
      const int64_t output_element =
          outer_idx * output_row_elements + row_element;
      CopyConcatElement(output, output_element, input, input_element,
                        element_size);
      return;
    }
    input_offset = input_end;
  }
}

__global__ void ConcatManySmallRowsKernel(
    uint8_t* output, const MusaConcatElementDesc* descriptors, int64_t outer,
    int64_t output_row_elements, int32_t element_size) {
  const int64_t outer_idx = static_cast<int64_t>(blockIdx.y);
  const int64_t row_element =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (outer_idx >= outer || row_element >= output_row_elements) {
    return;
  }

  const MusaConcatElementDesc desc = descriptors[row_element];
  const auto* input = static_cast<const uint8_t*>(desc.input);
  const int64_t input_element =
      outer_idx * desc.input_width + desc.local_element;
  const int64_t output_element = outer_idx * output_row_elements + row_element;

  CopyConcatElement(output, output_element, input, input_element, element_size);
}

}  // namespace

musaError_t LaunchMusaConcatBf16LastAxisHighArity(
    void* output, const void* const* inputs, const int64_t* input_axis_dims,
    int64_t input_count, int64_t outer, int64_t output_axis,
    musaStream_t stream) {
  constexpr int64_t kBf16ElementsPerVector = sizeof(uint4) / sizeof(uint16_t);
  if (output == nullptr || inputs == nullptr || input_axis_dims == nullptr ||
      input_count <= 0 || input_count > kMusaConcatBf16HighArityMaxInputs ||
      outer <= 0 || output_axis <= 0 ||
      output_axis % kBf16ElementsPerVector != 0 ||
      reinterpret_cast<uintptr_t>(output) % alignof(uint4) != 0) {
    return musaErrorInvalidValue;
  }

  MusaConcatBf16HighArityParams params{};
  params.outer = outer;
  params.output_row_vectors =
      static_cast<int32_t>(output_axis / kBf16ElementsPerVector);
  int32_t output_offset = 0;
  int64_t max_input_vectors = 0;
  for (int64_t i = 0; i < input_count; ++i) {
    if (inputs[i] == nullptr || input_axis_dims[i] <= 0 ||
        input_axis_dims[i] % kBf16ElementsPerVector != 0 ||
        reinterpret_cast<uintptr_t>(inputs[i]) % alignof(uint4) != 0) {
      return musaErrorInvalidValue;
    }
    params.inputs[i] = static_cast<const uint4*>(inputs[i]);
    params.input_row_vectors[i] =
        static_cast<int32_t>(input_axis_dims[i] / kBf16ElementsPerVector);
    params.output_offsets[i] = output_offset;
    output_offset += params.input_row_vectors[i];
    const int64_t input_vectors = outer * params.input_row_vectors[i];
    max_input_vectors =
        max_input_vectors > input_vectors ? max_input_vectors : input_vectors;
  }
  if (output_offset != params.output_row_vectors) {
    return musaErrorInvalidValue;
  }

  const unsigned int blocks = static_cast<unsigned int>(
      (max_input_vectors + kConcatCopyThreads - 1) / kConcatCopyThreads);
  dim3 grid(blocks, static_cast<unsigned int>(input_count));
  ConcatBf16LastAxisHighArityKernel<<<grid, kConcatCopyThreads, 0, stream>>>(
      static_cast<uint4*>(output), params);
  return musaGetLastError();
}

musaError_t LaunchMusaConcatCopies(void* output, const void* const* inputs,
                                   const int64_t* input_axis_dims,
                                   int64_t input_count, int64_t outer,
                                   int64_t inner, int64_t output_axis,
                                   int32_t element_size, musaStream_t stream) {
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

musaError_t LaunchMusaConcatManySmallRowsDirect(
    void* output, const void* const* inputs, const int64_t* input_axis_dims,
    int64_t input_count, int64_t outer, int64_t inner,
    int64_t output_row_elements, int32_t element_size, musaStream_t stream) {
  if (outer == 0 || output_row_elements == 0) {
    return musaSuccess;
  }
  if (input_count < 0 || input_count > kMusaConcatSmallRowsMaxInputs) {
    return musaErrorInvalidValue;
  }

  MusaConcatSmallRowsParams params{};
  params.input_count = input_count;
  params.inner = inner;
  for (int64_t i = 0; i < input_count; ++i) {
    params.inputs[i] = inputs[i];
    params.input_axis_dims[i] = input_axis_dims[i];
  }

  const auto blocks_per_row = static_cast<unsigned int>(
      (output_row_elements + kConcatCopyThreads - 1) / kConcatCopyThreads);
  dim3 grid(blocks_per_row, static_cast<unsigned int>(outer));
  ConcatManySmallRowsDirectKernel<<<grid, kConcatCopyThreads, 0, stream>>>(
      static_cast<uint8_t*>(output), params, outer, output_row_elements,
      element_size);
  return musaGetLastError();
}

musaError_t LaunchMusaConcatManySmallRows(
    void* output, const MusaConcatElementDesc* descriptors, int64_t outer,
    int64_t output_row_elements, int32_t element_size, musaStream_t stream) {
  if (outer == 0 || output_row_elements == 0) {
    return musaSuccess;
  }
  const auto blocks_per_row = static_cast<unsigned int>(
      (output_row_elements + kConcatCopyThreads - 1) / kConcatCopyThreads);
  dim3 grid(blocks_per_row, static_cast<unsigned int>(outer));
  ConcatManySmallRowsKernel<<<grid, kConcatCopyThreads, 0, stream>>>(
      static_cast<uint8_t*>(output), descriptors, outer, output_row_elements,
      element_size);
  return musaGetLastError();
}
