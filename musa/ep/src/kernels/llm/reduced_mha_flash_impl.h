#pragma once

#include <musa_runtime.h>

#include <cstdint>

// Packed QKV is [B,S,3*A].  This is an inference-only online-softmax kernel:
// it never allocates a [B,H,S,S] score/probability tensor.
struct MusaReducedMhaFlashParams {
  int64_t batch;
  int64_t sequence;
  int64_t attention_dim;
  int64_t heads;
  int64_t head_dim;
  int64_t mask_batch;
  int64_t mask_heads;
  float scale;
  bool has_mask;
};

musaError_t LaunchMusaReducedMhaFlashKernel(const float* packed_qkv,
                                            const int32_t* mask,
                                            float* attention_out,
                                            MusaReducedMhaFlashParams params,
                                            musaStream_t stream);
