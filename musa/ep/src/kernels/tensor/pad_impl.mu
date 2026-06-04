#include "tensor/pad_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ void WriteConstant(void* output,
                                              int64_t output_index,
                                              uint64_t constant_value,
                                              int32_t element_size) {
  if (element_size == 4) {
    reinterpret_cast<uint32_t*>(output)[output_index] =
        static_cast<uint32_t>(constant_value);
  } else if (element_size == 8) {
    reinterpret_cast<uint64_t*>(output)[output_index] = constant_value;
  } else if (element_size == 1) {
    reinterpret_cast<uint8_t*>(output)[output_index] =
        static_cast<uint8_t>(constant_value);
  } else {
    uint8_t* dst =
        reinterpret_cast<uint8_t*>(output) + output_index * element_size;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&constant_value);
    for (int32_t byte = 0; byte < element_size; ++byte) {
      dst[byte] = bytes[byte];
    }
  }
}

__device__ __forceinline__ void CopyOrFillPadElement(const void* input,
                                                     void* output,
                                                     int64_t input_index,
                                                     int64_t output_index,
                                                     uint64_t constant_value,
                                                     int32_t element_size,
                                                     bool inside) {
  if (!inside) {
    WriteConstant(output, output_index, constant_value, element_size);
    return;
  }

  if (element_size == 4) {
    reinterpret_cast<uint32_t*>(output)[output_index] =
        reinterpret_cast<const uint32_t*>(input)[input_index];
  } else if (element_size == 8) {
    reinterpret_cast<uint64_t*>(output)[output_index] =
        reinterpret_cast<const uint64_t*>(input)[input_index];
  } else if (element_size == 1) {
    reinterpret_cast<uint8_t*>(output)[output_index] =
        reinterpret_cast<const uint8_t*>(input)[input_index];
  } else {
    const uint8_t* src =
        reinterpret_cast<const uint8_t*>(input) + input_index * element_size;
    uint8_t* dst =
        reinterpret_cast<uint8_t*>(output) + output_index * element_size;
    for (int32_t byte = 0; byte < element_size; ++byte) {
      dst[byte] = src[byte];
    }
  }
}

__global__ void PadKernel(const void* input,
                          void* output,
                          uint64_t constant_value,
                          int32_t element_size,
                          MusaPadParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.total_elements;
       output_index += total_threads) {
    int64_t remaining = output_index;
    int64_t input_index = 0;
    bool inside = true;
    for (int32_t dim = params.rank - 1; dim >= 0; --dim) {
      const int64_t coord = remaining % params.output_dims[dim];
      remaining /= params.output_dims[dim];
      const int64_t input_coord = coord - params.pads_begin[dim];
      if (input_coord < 0 || input_coord >= params.input_dims[dim]) {
        inside = false;
      } else {
        input_index += input_coord * params.input_strides[dim];
      }
    }
    CopyOrFillPadElement(input, output, input_index, output_index,
                         constant_value, element_size, inside);
  }
}

}  // namespace

musaError_t LaunchMusaPadKernel(const void* input,
                                void* output,
                                uint64_t constant_value,
                                int32_t element_size,
                                MusaPadParams params,
                                musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  PadKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0,
              stream>>>(input, output, constant_value, element_size, params);
  return musaGetLastError();
}
