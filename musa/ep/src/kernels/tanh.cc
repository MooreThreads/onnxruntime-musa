// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Tanh : public OpKernelBase<Tanh> {
 public:
  Tanh(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Tanh::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "unsupported unary op dtype");
  }
  return UnaryCompute<float>(ctx, info.GetShape(),
                             [](float x) { return std::tanh(x); },
                             MusaUnaryOp::Tanh);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Tanh, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())), Tanh)
