// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class LeakyRelu : public OpKernelBase<LeakyRelu> {
 public:
  LeakyRelu(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    alpha_ = AttrOrDefault<float>(kernel_info, "alpha", 0.01f);
    alpha_ = AttrOrDefault<float>(kernel_info, "activation_alpha", alpha_);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  float alpha_ = 0.01f;
};

OrtStatus* LeakyRelu::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "unsupported unary op dtype");
  }
  float alpha = alpha_;
  return UnaryCompute<float>(ctx, info.GetShape(), [alpha](float x) {
    return x >= 0.0f ? x : alpha * x;
  });
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    LeakyRelu, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    LeakyRelu)
