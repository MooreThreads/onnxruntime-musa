#pragma once

#include "shared_inc/device_kernel_types.h"

struct MusaBatchedMatMulParams {
  int32_t output_rank;
  int32_t batch_rank;
  int64_t total_elements;
  int64_t m;
  int64_t n;
  int64_t k;
  float alpha;
  int64_t output_dims[kMusaMaxBroadcastRank];
  int64_t output_strides[kMusaMaxBroadcastRank];
  int64_t a_batch_strides[kMusaMaxBroadcastRank];
  int64_t b_batch_strides[kMusaMaxBroadcastRank];
  int64_t a_row_stride;
  int64_t a_col_stride;
  int64_t b_row_stride;
  int64_t b_col_stride;
};

musaError_t LaunchMusaBatchedMatMulKernel(const void* a, const void* b,
                                          void* output,
                                          MusaBatchedMatMulParams params,
                                          MusaElementType elem_type,
                                          musaStream_t stream);
