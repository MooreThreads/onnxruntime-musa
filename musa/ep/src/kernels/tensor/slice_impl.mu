#include "tensor/slice_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void SliceKernel(const void* input,
                            void* output,
                            int32_t element_size,
                            MusaSliceParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.total_elements; output_index += total_threads) {
    int64_t remaining = output_index;
    int64_t input_index = 0;
    for (int32_t dim = params.rank - 1; dim >= 0; --dim) {
      const int64_t coord = remaining % params.output_dims[dim];
      remaining /= params.output_dims[dim];
      input_index += (params.starts[dim] + coord * params.steps[dim]) * params.input_strides[dim];
    }

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

template <typename T>
__global__ void SliceLastAxisKernel(const T* input,
                                    T* output,
                                    int64_t total_elements,
                                    int64_t input_width,
                                    int64_t output_width,
                                    int64_t start) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < total_elements;
       output_index += total_threads) {
    const int64_t outer = output_index / output_width;
    const int64_t inner = output_index - outer * output_width;
    output[output_index] = input[outer * input_width + start + inner];
  }
}

__global__ void SliceLastAxisFloat4Kernel(const float* input,
                                          float* output,
                                          int64_t total_chunks,
                                          int64_t input_width,
                                          int64_t output_width,
                                          int64_t start) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t chunks_per_row = output_width >> 2;

  for (int64_t chunk_index = thread_id; chunk_index < total_chunks;
       chunk_index += total_threads) {
    const int64_t outer = chunk_index / chunks_per_row;
    const int64_t inner_chunk = chunk_index - outer * chunks_per_row;
    const int64_t inner = inner_chunk << 2;
    const int64_t input_offset = outer * input_width + start + inner;
    const int64_t output_offset = outer * output_width + inner;
    output[output_offset] = input[input_offset];
    output[output_offset + 1] = input[input_offset + 1];
    output[output_offset + 2] = input[input_offset + 2];
    output[output_offset + 3] = input[input_offset + 3];
  }
}

template <typename T>
musaError_t LaunchSliceLastAxisTyped(const void* input,
                                     void* output,
                                     int64_t total_elements,
                                     int64_t input_width,
                                     int64_t output_width,
                                     int64_t start,
                                     musaStream_t stream) {
  SliceLastAxisKernel<T><<<BlocksForCount(total_elements), kThreadsPerBlock, 0,
                           stream>>>(reinterpret_cast<const T*>(input),
                                     reinterpret_cast<T*>(output),
                                     total_elements, input_width, output_width,
                                     start);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaSliceKernel(const void* input,
                                  void* output,
                                  int32_t element_size,
                                  MusaSliceParams params,
                                  musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  SliceKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(input, output, element_size, params);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}

musaError_t LaunchMusaSliceLastAxisKernel(
    const void* input, void* output, int32_t element_size,
    int64_t total_elements, int64_t input_width, int64_t output_width,
    int64_t start, musaStream_t stream) {
  if (total_elements == 0) {
    return musaSuccess;
  }
  if (element_size == 4 && output_width > 0 && output_width % 4 == 0) {
    const int64_t total_chunks = total_elements >> 2;
    SliceLastAxisFloat4Kernel<<<BlocksForCount(total_chunks), kThreadsPerBlock,
                                0, stream>>>(
        reinterpret_cast<const float*>(input), reinterpret_cast<float*>(output),
        total_chunks, input_width, output_width, start);
    return musaGetLastError();
  }
  switch (element_size) {
    case 1:
      return LaunchSliceLastAxisTyped<uint8_t>(
          input, output, total_elements, input_width, output_width, start,
          stream);
    case 2:
      return LaunchSliceLastAxisTyped<uint16_t>(
          input, output, total_elements, input_width, output_width, start,
          stream);
    case 4:
      return LaunchSliceLastAxisTyped<uint32_t>(
          input, output, total_elements, input_width, output_width, start,
          stream);
    case 8:
      return LaunchSliceLastAxisTyped<uint64_t>(
          input, output, total_elements, input_width, output_width, start,
          stream);
    default:
      return musaErrorNotSupported;
  }
}
