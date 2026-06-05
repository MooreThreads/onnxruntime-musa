#include "shared_inc/musa_kernel_common.mu.h"
#include "tensor/reverse_sequence_impl.h"

namespace {

__device__ __forceinline__ void CopyReverseSequenceElement(
    const void* input, void* output, int64_t input_index, int64_t output_index,
    int32_t element_size) {
  if (element_size == 8) {
    reinterpret_cast<uint64_t*>(output)[output_index] =
        reinterpret_cast<const uint64_t*>(input)[input_index];
  } else if (element_size == 4) {
    reinterpret_cast<uint32_t*>(output)[output_index] =
        reinterpret_cast<const uint32_t*>(input)[input_index];
  } else if (element_size == 2) {
    reinterpret_cast<uint16_t*>(output)[output_index] =
        reinterpret_cast<const uint16_t*>(input)[input_index];
  } else if (element_size == 1) {
    reinterpret_cast<uint8_t*>(output)[output_index] =
        reinterpret_cast<const uint8_t*>(input)[input_index];
  } else {
    const auto* src =
        reinterpret_cast<const uint8_t*>(input) + input_index * element_size;
    auto* dst =
        reinterpret_cast<uint8_t*>(output) + output_index * element_size;
    for (int32_t byte = 0; byte < element_size; ++byte) {
      dst[byte] = src[byte];
    }
  }
}

__global__ void ReverseSequenceKernel(const void* input,
                                      const int64_t* sequence_lens,
                                      void* output, int32_t element_size,
                                      MusaReverseSequenceParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.total_elements;
       output_index += total_threads) {
    const int64_t element_id = output_index % params.element_size;
    int64_t grouped = output_index / params.element_size;

    int64_t seq_id = 0;
    int64_t batch_id = 0;
    if (params.time_major) {
      batch_id = grouped % params.batch_size;
      seq_id = grouped / params.batch_size;
    } else {
      seq_id = grouped % params.max_seq_len;
      batch_id = grouped / params.max_seq_len;
    }

    int64_t seq_len = sequence_lens[batch_id];
    if (seq_len < 0) {
      seq_len = 0;
    }
    if (seq_len > params.max_seq_len) {
      seq_len = params.max_seq_len;
    }
    const int64_t target_seq_id =
        seq_id < seq_len ? seq_len - 1 - seq_id : seq_id;

    int64_t input_index = 0;
    if (params.time_major) {
      input_index =
          (target_seq_id * params.batch_size + batch_id) * params.element_size +
          element_id;
    } else {
      input_index = (batch_id * params.max_seq_len + target_seq_id) *
                        params.element_size +
                    element_id;
    }

    CopyReverseSequenceElement(input, output, input_index, output_index,
                               element_size);
  }
}

}  // namespace

musaError_t LaunchMusaReverseSequenceKernel(const void* input,
                                            const int64_t* sequence_lens,
                                            void* output, int32_t element_size,
                                            MusaReverseSequenceParams params,
                                            musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  ReverseSequenceKernel<<<BlocksForCount(params.total_elements),
                          kThreadsPerBlock, 0, stream>>>(
      input, sequence_lens, output, element_size, params);
  return musaGetLastError();
}
