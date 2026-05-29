#pragma once

#include "utils.h"

#include <string>
#include <vector>

class BasicOp : public OrtKernelImpl {
 private:
  struct PrivateTag {};

 public:
  static OrtStatus* CreateKernelImpl(const OrtKernelInfo* info, void* state, OrtKernelImpl*& kernel) noexcept;
  BasicOp(const OrtKernelInfo* info, void* state, PrivateTag);

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  OrtStatus* ComputeInternal(Ort::KernelContext& ctx) const;

  std::string op_type_;
  int64_t axis_ = 0;
  int64_t keepdims_ = 1;
  int64_t allowzero_ = 0;
  int64_t to_ = 0;
  int64_t trans_a_ = 0;
  int64_t trans_b_ = 0;
  int64_t trans_batch_a_ = 0;
  int64_t trans_batch_b_ = 0;
  float alpha_gemm_ = 1.0f;
  float beta_ = 1.0f;
  float alpha_ = 0.01f;
  std::string activation_;
  std::vector<int64_t> axes_attr_;
  std::vector<int64_t> perm_attr_;
};
