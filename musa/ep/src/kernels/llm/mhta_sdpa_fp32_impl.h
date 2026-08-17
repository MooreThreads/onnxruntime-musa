#pragma once

#include <musa_runtime.h>

#include <cstdint>

#include "shared_inc/device_kernel_types.h"

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
  // When true, mask is a BOOL keep-mask and invalid keys receive -inf.
  // Otherwise mask is a FLOAT additive mask scaled by mask_scale.
  bool boolean_mask;
  // ranking-gr builds its BOOL keep-mask as Equal(Cast(Slice(INT32)), 1).
  // The fused path reads the unsliced INT32 mask directly and applies the
  // same equality while limiting columns to seqlen_k.
  bool boolean_mask_int32;
};

musaError_t LaunchMusaMhtaSdpaFp32Kernel(const float* q, const float* k,
                                         const float* v, const void* mask,
                                         float* output,
                                         MusaMhtaSdpaFp32Params params,
                                         musaStream_t stream);

musaError_t LaunchMusaMhtaSdpaKeepMaskToAdditiveKernel(
    const void* mask, void* additive_mask, MusaMhtaSdpaFp32Params params,
    MusaElementType output_elem_type, musaStream_t stream);
