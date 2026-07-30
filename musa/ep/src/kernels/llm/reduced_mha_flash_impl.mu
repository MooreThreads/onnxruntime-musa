#include "llm/reduced_mha_flash_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

// For the common head_dim=64 path, 256 threads leave three quarters of the
// block idle while accumulating P*V.  128 threads still cover a score row in
// a few iterations and permits more resident blocks for the many short rows
// in decoder attention.
constexpr int kThreads = 64;
constexpr int64_t kMaxKeys = 8192;
constexpr float kNegInf = -3.4028234663852886e38f;

__device__ __forceinline__ int64_t
MaskOffset(const MusaReducedMhaFlashParams& p, int64_t b, int64_t h, int64_t q,
           int64_t k) {
  const int64_t mb = p.mask_batch == 1 ? 0 : b;
  const int64_t mh = p.mask_heads == 1 ? 0 : h;
  return ((mb * p.mask_heads + mh) * p.sequence + q) * p.sequence + k;
}

__global__ void ReducedMhaFlashKernel(const float* qkv, const int32_t* mask,
                                      float* out, MusaReducedMhaFlashParams p) {
  const int64_t row_id = static_cast<int64_t>(blockIdx.x);
  const int64_t rows = p.batch * p.heads * p.sequence;
  if (row_id >= rows) return;
  const int64_t q_pos = row_id % p.sequence;
  const int64_t head_index = row_id / p.sequence;
  const int64_t h = head_index % p.heads;
  const int64_t b = head_index / p.heads;

  extern __shared__ float shared[];
  float* scores = shared;
  float* reduce = shared + p.sequence;
  const int64_t q_base =
      (b * p.sequence + q_pos) * (3 * p.attention_dim) + h * p.head_dim;

  float local_max = kNegInf;
  for (int64_t k_pos = threadIdx.x; k_pos < p.sequence; k_pos += blockDim.x) {
    float score = kNegInf;
    const int32_t mask_value =
        p.has_mask ? mask[MaskOffset(p, b, h, q_pos, k_pos)] : 1;
    const bool keep = !p.has_mask ||
                      (p.mask_positive_only ? mask_value > 0 : mask_value != 0);
    if (keep || p.use_mask_filter_value) {
      const int64_t k_base = (b * p.sequence + k_pos) * (3 * p.attention_dim) +
                             p.attention_dim + h * p.head_dim;
      float dot = 0.0f;
      // All supported packed-QKV offsets are 16-byte aligned for head_dim 64.
      // Vector loads reduce instruction and address-generation overhead in
      // the QK hot loop.  Keep the scalar tail for generic head dimensions.
      int64_t d = 0;
      if (p.head_dim % 4 == 0) {
        for (; d + 3 < p.head_dim; d += 4) {
          const float4 qv = *reinterpret_cast<const float4*>(qkv + q_base + d);
          const float4 kv = *reinterpret_cast<const float4*>(qkv + k_base + d);
          dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        }
      }
      for (; d < p.head_dim; ++d) dot += qkv[q_base + d] * qkv[k_base + d];
      score = dot * p.scale + (keep ? 0.0f : p.mask_filter_value);
    }
    scores[k_pos] = score;
    local_max = fmaxf(local_max, score);
  }
  reduce[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      reduce[threadIdx.x] =
          fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
    __syncthreads();
  }
  const float max_score = reduce[0];
  float local_sum = 0.0f;
  for (int64_t k_pos = threadIdx.x; k_pos < p.sequence; k_pos += blockDim.x) {
    const float weight =
        scores[k_pos] == kNegInf ? 0.0f : expf(scores[k_pos] - max_score);
    scores[k_pos] = weight;
    local_sum += weight;
  }
  reduce[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_sum = reduce[0] == 0.0f ? 0.0f : 1.0f / reduce[0];
  const int64_t out_base =
      (b * p.sequence + q_pos) * p.attention_dim + h * p.head_dim;
  for (int64_t d = threadIdx.x; d < p.head_dim; d += blockDim.x) {
    float value = 0.0f;
    for (int64_t k_pos = 0; k_pos < p.sequence; ++k_pos) {
      const int64_t v_base = (b * p.sequence + k_pos) * (3 * p.attention_dim) +
                             2 * p.attention_dim + h * p.head_dim;
      value += scores[k_pos] * qkv[v_base + d];
    }
    out[out_base + d] = value * inv_sum;
  }
}

}  // namespace

musaError_t LaunchMusaReducedMhaFlashKernel(const float* packed_qkv,
                                            const int32_t* mask,
                                            float* attention_out,
                                            MusaReducedMhaFlashParams params,
                                            musaStream_t stream) {
  if (params.batch <= 0 || params.sequence <= 0 || params.attention_dim <= 0 ||
      params.heads <= 0 || params.head_dim <= 0)
    return musaSuccess;
  if (params.sequence > kMaxKeys ||
      params.batch * params.heads * params.sequence > INT32_MAX)
    return musaErrorNotSupported;
  const size_t shared_bytes =
      static_cast<size_t>(params.sequence + kThreads) * sizeof(float);
  const int64_t rows = params.batch * params.heads * params.sequence;
  ReducedMhaFlashKernel<<<static_cast<unsigned int>(rows), kThreads,
                          shared_bytes, stream>>>(packed_qkv, mask,
                                                  attention_out, params);
  return musaGetLastError();
}
