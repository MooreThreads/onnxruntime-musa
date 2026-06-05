// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reduction/reduction_utils.h"

namespace {
class ReduceL2 : public OpKernelBase<ReduceL2> {
 public:
  ReduceL2(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    keepdims_ = AttrOrDefault<int64_t>(kernel_info, "keepdims", 1);
    axes_attr_ = AttrsOrEmpty(kernel_info, "axes");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const {
    return ReduceCompute(ctx, axes_attr_, keepdims_ != 0, ReduceMode::kL2);
  }

 private:
  int64_t keepdims_ = 1;
  std::vector<int64_t> axes_attr_;
};
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ReduceL2, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", ReduceL2Opset13TensorTypes())
         .SetInputMemType(1, OrtMemTypeCPUInput)),
    ReduceL2)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ReduceL2, kOnnxDomain, 18, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", ReduceMeanTensorTypes())
         .SetInputMemType(1, OrtMemTypeCPUInput)),
    ReduceL2)
