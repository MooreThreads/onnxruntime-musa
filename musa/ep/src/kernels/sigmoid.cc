// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Sigmoid : public OpKernelBase<Sigmoid> {
 public:
  Sigmoid(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Sigmoid::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "unsupported unary op dtype");
  }
  return UnaryCompute<float>(ctx, info.GetShape(), [](float x) {
    return 1.0f / (1.0f + std::exp(-x));
  });
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sigmoid, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    Sigmoid)
