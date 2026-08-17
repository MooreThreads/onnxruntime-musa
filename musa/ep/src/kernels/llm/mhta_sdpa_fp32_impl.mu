#include "llm/mhta_sdpa_fp32_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

constexpr int kMhtaSdpaThreads = 256;
// Scores remain on chip, as in FlashAttention.  This bound keeps the dynamic
// shared-memory allocation below the portable 48 KiB launch limit.
constexpr int64_t kMaxMhtaSdpaKeys = 8192;

__device__ __forceinline__ int64_t
MaskOffset(const MusaMhtaSdpaFp32Params& params, int64_t b, int64_t h,
           int64_t q, int64_t k) {
  const int64_t mask_b = params.mask_b == 1 ? 0 : b;
  const int64_t mask_h = params.mask_h == 1 ? 0 : h;
  const int64_t mask_q = params.mask_q == 1 ? 0 : q;
  const int64_t mask_k = params.mask_k == 1 ? 0 : k;
  return ((mask_b * params.mask_h + mask_h) * params.mask_q + mask_q) *
             params.mask_k +
         mask_k;
}

__device__ __forceinline__ int64_t AdditiveMaskK(
    const MusaMhtaSdpaFp32Params& params) {
  return params.boolean_mask_int32 ? params.seqlen_k : params.mask_k;
}

template <typename T>
__device__ __forceinline__ T MhtaSdpaMaskValue(bool keep) {
  return MusaScalarFromFloat<T>(keep ? 0.0f : -INFINITY);
}

__device__ __forceinline__ float Score(const float* q, const float* k,
                                       const void* mask,
                                       const MusaMhtaSdpaFp32Params& params,
                                       int64_t b, int64_t h, int64_t row,
                                       int64_t col) {
  const int64_t q_base =
      params.sim_rank3
          ? ((b * params.seqlen_q + row) * params.heads + h) * params.head_dim
          : ((b * params.heads + h) * params.seqlen_q + row) * params.head_dim;
  const int64_t k_head_base =
      (b * params.heads + h) * params.seqlen_k * params.head_dim;
  float dot = 0.0f;
  for (int64_t d = 0; d < params.head_dim; ++d) {
    const int64_t k_offset =
        params.sim_rank3
            ? ((b * params.seqlen_k + col) * params.heads + h) *
                      params.head_dim +
                  d
            : (params.key_is_bhds ? (k_head_base + d * params.seqlen_k + col)
                                  : (k_head_base + col * params.head_dim + d));
    dot += q[q_base + d] * k[k_offset];
  }
  const int64_t mask_offset = MaskOffset(params, b, h, row, col);
  if (params.boolean_mask) {
    const bool keep = params.boolean_mask_int32
                          ? static_cast<const int32_t*>(mask)[mask_offset] == 1
                          : static_cast<const uint8_t*>(mask)[mask_offset] != 0;
    return keep ? dot * params.scale : -3.4028234663852886e38f;
  }
  return dot * params.scale +
         static_cast<const float*>(mask)[mask_offset] * params.mask_scale;
}

template <typename T>
__global__ void MhtaKeepMaskToAdditiveKernel(
    const void* mask, T* additive_mask, MusaMhtaSdpaFp32Params params) {
  const int64_t out_mask_k = AdditiveMaskK(params);
  const int64_t count =
      params.mask_b * params.mask_h * params.mask_q * out_mask_k;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; index < count; index += stride) {
    const int64_t k = index % out_mask_k;
    const int64_t q_index = index / out_mask_k;
    const int64_t q = q_index % params.mask_q;
    const int64_t h_index = q_index / params.mask_q;
    const int64_t h = h_index % params.mask_h;
    const int64_t b = h_index / params.mask_h;
    const int64_t mask_offset = MaskOffset(params, b, h, q, k);
    const bool keep =
        params.boolean_mask_int32
            ? static_cast<const int32_t*>(mask)[mask_offset] == 1
            : static_cast<const uint8_t*>(mask)[mask_offset] != 0;
    additive_mask[index] = MhtaSdpaMaskValue<T>(keep);
  }
}

__global__ void MhtaSdpaFp32Kernel(const float* q, const float* k,
                                   const float* v, const void* mask,
                                   float* output,
                                   MusaMhtaSdpaFp32Params params) {
  const int64_t row_id = static_cast<int64_t>(blockIdx.x);
  const int64_t rows = params.batch * params.heads * params.seqlen_q;
  if (row_id >= rows) {
    return;
  }
  const int64_t row = row_id % params.seqlen_q;
  const int64_t head_index = row_id / params.seqlen_q;
  const int64_t h = head_index % params.heads;
  const int64_t b = head_index / params.heads;

  extern __shared__ float shared[];
  float* scores = shared;
  float* reduce = shared + params.seqlen_k;

  float local_max = -3.4028234663852886e38f;
  for (int64_t col = threadIdx.x; col < params.seqlen_k; col += blockDim.x) {
    const float score = Score(q, k, mask, params, b, h, row, col);
    scores[col] = score;
    local_max = fmaxf(local_max, score);
  }
  reduce[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce[threadIdx.x] =
          fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const float max_score = reduce[0];
  float local_sum = 0.0f;
  for (int64_t col = threadIdx.x; col < params.seqlen_k; col += blockDim.x) {
    const float weight = scores[col] == -3.4028234663852886e38f
                             ? 0.0f
                             : expf(scores[col] - max_score);
    scores[col] = weight;
    local_sum += weight;
  }
  reduce[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_sum = reduce[0] == 0.0f ? 0.0f : 1.0f / reduce[0];
  const int64_t output_base =
      ((b * params.heads + h) * params.seqlen_q + row) * params.head_dim;
  const int64_t v_head_base =
      (b * params.heads + h) * params.seqlen_k * params.head_dim;
  for (int64_t d = threadIdx.x; d < params.head_dim; d += blockDim.x) {
    float value = 0.0f;
    for (int64_t col = 0; col < params.seqlen_k; ++col) {
      value += scores[col] * v[v_head_base + col * params.head_dim + d];
    }
    output[output_base + d] = value * inv_sum;
  }
}

}  // namespace

musaError_t LaunchMusaMhtaSdpaFp32Kernel(const float* q, const float* k,
                                         const float* v, const void* mask,
                                         float* output,
                                         MusaMhtaSdpaFp32Params params,
                                         musaStream_t stream) {
  if (params.batch <= 0 || params.heads <= 0 || params.seqlen_q <= 0 ||
      params.seqlen_k <= 0 || params.head_dim <= 0) {
    return musaSuccess;
  }
  if (params.seqlen_k > kMaxMhtaSdpaKeys ||
      params.batch * params.heads * params.seqlen_q > INT32_MAX) {
    return musaErrorNotSupported;
  }
  const size_t shared_bytes =
      static_cast<size_t>(params.seqlen_k + kMhtaSdpaThreads) * sizeof(float);
  const int64_t rows = params.batch * params.heads * params.seqlen_q;
  MhtaSdpaFp32Kernel<<<static_cast<unsigned int>(rows), kMhtaSdpaThreads,
                       shared_bytes, stream>>>(q, k, v, mask, output, params);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchMusaMhtaSdpaKeepMaskToAdditiveTyped(
    const void* mask, void* additive_mask, MusaMhtaSdpaFp32Params params,
    musaStream_t stream) {
  const int64_t out_mask_k = params.boolean_mask_int32 ? params.seqlen_k
                                                       : params.mask_k;
  if (params.mask_b <= 0 || params.mask_h <= 0 || params.mask_q <= 0 ||
      out_mask_k <= 0) {
    return musaSuccess;
  }
  const int64_t count =
      params.mask_b * params.mask_h * params.mask_q * out_mask_k;
  MhtaKeepMaskToAdditiveKernel<<<BlocksForCount(count), kThreadsPerBlock, 0,
                                 stream>>>(
      mask, static_cast<T*>(additive_mask), params);
  return musaGetLastError();
}

musaError_t LaunchMusaMhtaSdpaKeepMaskToAdditiveKernel(
    const void* mask, void* additive_mask, MusaMhtaSdpaFp32Params params,
    MusaElementType output_elem_type, musaStream_t stream) {
  if (!params.boolean_mask) {
    return musaErrorInvalidValue;
  }
  switch (output_elem_type) {
    case MusaElementType::Float16:
      return LaunchMusaMhtaSdpaKeepMaskToAdditiveTyped<__half>(
          mask, additive_mask, params, stream);
    case MusaElementType::BFloat16:
      return LaunchMusaMhtaSdpaKeepMaskToAdditiveTyped<__mt_bfloat16>(
          mask, additive_mask, params, stream);
    default:
      return musaErrorNotSupported;
  }
}
