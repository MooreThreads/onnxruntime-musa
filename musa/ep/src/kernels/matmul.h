#pragma once

#include "utils.h"

#include <cstdint>
#include <vector>

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
