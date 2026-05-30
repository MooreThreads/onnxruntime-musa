// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/reduce_utils.h"

namespace {
class ReduceSum : public OpKernelBase<ReduceSum> {
 public:
  ReduceSum(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    keepdims_ = AttrOrDefault<int64_t>(kernel_info, "keepdims", 1);
    axes_attr_ = AttrsOrEmpty(kernel_info, "axes");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const {
    return ReduceCompute(ctx, axes_attr_, keepdims_ != 0, ReduceMode::kSum);
  }

 private:
  int64_t keepdims_ = 1;
  std::vector<int64_t> axes_attr_;
};
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ReduceSum, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())),
    ReduceSum)
