#include "llm/multihead_attention_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void PackQkvBiasKernel(const float* query, const float* key,
                                  const float* value, const float* bias,
                                  float* packed_qkv, int64_t element_count,
                                  int64_t hidden_size) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t thread_count = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < element_count;
       index += thread_count) {
    const int64_t hidden = index % hidden_size;
    const int64_t packed_base =
        (index / hidden_size) * (3 * hidden_size) + hidden;
    packed_qkv[packed_base] = query[index] + bias[hidden];
    packed_qkv[packed_base + hidden_size] =
        key[index] + bias[hidden_size + hidden];
    packed_qkv[packed_base + 2 * hidden_size] =
        value[index] + bias[2 * hidden_size + hidden];
  }
}

}  // namespace

musaError_t LaunchMusaPackQkvBiasKernel(const float* query, const float* key,
                                        const float* value, const float* bias,
                                        float* packed_qkv, int64_t token_count,
                                        int64_t hidden_size,
                                        musaStream_t stream) {
  if (token_count <= 0 || hidden_size <= 0) {
    return musaSuccess;
  }
  const int64_t element_count = token_count * hidden_size;
  PackQkvBiasKernel<<<BlocksForCount(element_count), kThreadsPerBlock, 0,
                      stream>>>(query, key, value, bias, packed_qkv,
                                element_count, hidden_size);
  return musaGetLastError();
}
