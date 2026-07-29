#pragma once

#include <musa_runtime.h>

#include <cstdint>

// The MatMul graph uses Q/V BHSD and K BHDS. The SIM Einsum graph preserves
// its Q[B,1,H,D] and K[B,S,H,D] inputs. Mask dimensions are padded on the left
// to B/H/Q/K and therefore describe ordinary ONNX broadcasting.
struct MusaMhtaSdpaFp32Params {
  int64_t batch;
  int64_t heads;
  int64_t seqlen_q;
  int64_t seqlen_k;
  int64_t head_dim;
  float scale;
  float mask_scale;
  int64_t mask_b;
  int64_t mask_h;
  int64_t mask_q;
  int64_t mask_k;
  bool key_is_bhds;
  bool sim_rank3;
};

musaError_t LaunchMusaMhtaSdpaFp32Kernel(const float* q, const float* k,
                                         const float* v, const float* mask,
                                         float* output,
                                         MusaMhtaSdpaFp32Params params,
                                         musaStream_t stream);
