#include "llm/attention_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

constexpr float kAttentionMaskValue = -3.4028234663852886e38f;

__global__ void AttentionAddBiasKernel(float* qkv, const float* bias,
                                       int64_t total_elements,
                                       int64_t qkv_hidden_size) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = thread_id; i < total_elements; i += total_threads) {
    qkv[i] += bias[i % qkv_hidden_size];
  }
}

__device__ __forceinline__ int64_t QkvOffset(const MusaAttentionParams& params,
                                             int64_t batch, int64_t seq,
                                             int64_t hidden_offset) {
  return (batch * params.sequence_length + seq) * params.qkv_hidden_size +
         hidden_offset;
}

__device__ __forceinline__ int32_t ReadMask(const int32_t* mask,
                                            const MusaAttentionParams& params,
                                            int64_t batch, int64_t head,
                                            int64_t query, int64_t key) {
  if (params.has_mask == 0 || mask == nullptr) {
    return 1;
  }
  const int64_t mask_b = params.mask_batch == 1 ? 0 : batch;
  const int64_t mask_h = params.mask_heads == 1 ? 0 : head;
  const int64_t offset =
      ((mask_b * params.mask_heads + mask_h) * params.sequence_length + query) *
          params.sequence_length +
      key;
  return mask[offset];
}

__global__ void AttentionScoreKernel(const float* qkv, const int32_t* mask,
                                     float* scores,
                                     MusaAttentionParams params) {
  const int64_t score_count = params.batch_size * params.num_heads *
                              params.sequence_length * params.sequence_length;
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < score_count; index += total_threads) {
    int64_t remaining = index;
    const int64_t key = remaining % params.sequence_length;
    remaining /= params.sequence_length;
    const int64_t query = remaining % params.sequence_length;
    remaining /= params.sequence_length;
    const int64_t head = remaining % params.num_heads;
    const int64_t batch = remaining / params.num_heads;

    if (ReadMask(mask, params, batch, head, query, key) == 0) {
      scores[index] = kAttentionMaskValue;
      continue;
    }

    float sum = 0.0f;
    const int64_t q_base = head * params.q_head_size;
    const int64_t k_base = params.q_hidden_size + head * params.k_head_size;
    for (int64_t d = 0; d < params.q_head_size; ++d) {
      const float q = qkv[QkvOffset(params, batch, query, q_base + d)];
      const float k = qkv[QkvOffset(params, batch, key, k_base + d)];
      sum += q * k;
    }
    scores[index] = sum * params.scale;
  }
}

__global__ void AttentionValueKernel(const float* qkv, const float* scores,
                                     float* output,
                                     MusaAttentionParams params) {
  const int64_t output_count =
      params.batch_size * params.sequence_length * params.v_hidden_size;
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < output_count; index += total_threads) {
    int64_t remaining = index;
    const int64_t v_dim = remaining % params.v_hidden_size;
    remaining /= params.v_hidden_size;
    const int64_t query = remaining % params.sequence_length;
    const int64_t batch = remaining / params.sequence_length;
    const int64_t head = v_dim / params.v_head_size;
    const int64_t head_dim = v_dim % params.v_head_size;

    const int64_t score_base =
        ((batch * params.num_heads + head) * params.sequence_length + query) *
        params.sequence_length;

    float max_score = kAttentionMaskValue;
    for (int64_t key = 0; key < params.sequence_length; ++key) {
      const float score = scores[score_base + key];
      max_score = fmaxf(max_score, score);
    }

    if (max_score == kAttentionMaskValue) {
      output[index] = 0.0f;
      continue;
    }

    float denom = 0.0f;
    float weighted = 0.0f;
    const int64_t v_base = params.q_hidden_size + params.k_hidden_size +
                           head * params.v_head_size + head_dim;
    for (int64_t key = 0; key < params.sequence_length; ++key) {
      const float score = scores[score_base + key];
      if (score == kAttentionMaskValue) {
        continue;
      }
      const float weight = expf(score - max_score);
      denom += weight;
      weighted += weight * qkv[QkvOffset(params, batch, key, v_base)];
    }
    output[index] = denom == 0.0f ? 0.0f : weighted / denom;
  }
}

}  // namespace

musaError_t LaunchMusaAttentionAddBiasKernel(float* qkv, const float* bias,
                                             int64_t total_elements,
                                             int64_t qkv_hidden_size,
                                             musaStream_t stream) {
  if (total_elements == 0) {
    return musaSuccess;
  }
  AttentionAddBiasKernel<<<BlocksForCount(total_elements), kThreadsPerBlock, 0,
                           stream>>>(qkv, bias, total_elements,
                                      qkv_hidden_size);
  return musaGetLastError();
}

musaError_t LaunchMusaAttentionScoreKernel(const float* qkv,
                                           const int32_t* mask, float* scores,
                                           MusaAttentionParams params,
                                           musaStream_t stream) {
  const int64_t score_count = params.batch_size * params.num_heads *
                              params.sequence_length * params.sequence_length;
  if (score_count == 0) {
    return musaSuccess;
  }
  AttentionScoreKernel<<<BlocksForCount(score_count), kThreadsPerBlock, 0,
                         stream>>>(qkv, mask, scores, params);
  return musaGetLastError();
}

musaError_t LaunchMusaAttentionValueKernel(const float* qkv,
                                           const float* scores, float* output,
                                           MusaAttentionParams params,
                                           musaStream_t stream) {
  const int64_t output_count =
      params.batch_size * params.sequence_length * params.v_hidden_size;
  if (output_count == 0) {
    return musaSuccess;
  }
  AttentionValueKernel<<<BlocksForCount(output_count), kThreadsPerBlock, 0,
                         stream>>>(qkv, scores, output, params);
  return musaGetLastError();
}
