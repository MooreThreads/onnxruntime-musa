#pragma once

#include "utils.h"

#include <cstdint>
#include <vector>

OrtStatus* ComputeMusaMatMulOutputShape(
    const std::vector<int64_t>& a_shape,
    const std::vector<int64_t>& b_shape,
    bool trans_a,
    bool trans_b,
    bool trans_batch_a,
    bool trans_batch_b,
    std::vector<int64_t>& y_shape);

OrtStatus* ComputeMusaMatMulDevice(const void* a_data, const void* b_data,
                                   void* y_data,
                                   ONNXTensorElementDataType elem_type,
                                   const std::vector<int64_t>& a_shape,
                                   const std::vector<int64_t>& b_shape,
                                   const std::vector<int64_t>& y_shape,
                                   bool trans_a = false,
                                   bool trans_b = false,
                                   bool trans_batch_a = false,
                                   bool trans_batch_b = false,
                                   float alpha = 1.0f);

OrtStatus* ComputeMusaMatMulDevice(const float* a_data, const float* b_data,
                                   float* y_data,
                                   const std::vector<int64_t>& a_shape,
                                   const std::vector<int64_t>& b_shape,
                                   const std::vector<int64_t>& y_shape);

struct MatMul {
  struct PrivateTag {};

  MatMul(const OrtKernelInfo* info, void* state, PrivateTag);

  static OrtStatus* CreateKernelImpl(const OrtKernelInfo* info, void* state, OrtKernelImpl*& kernel) noexcept;
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

  OrtKernelImpl kernel_base;
  const OrtKernelInfo* info_;
};
