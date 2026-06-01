// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Abs : public OpKernelBase<Abs> {
 public:
  Abs(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Abs::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Abs only supports float tensors");
  }
  return UnaryCompute<float>(ctx, info.GetShape(),
                             [](float x) { return std::fabs(x); },
                             MusaUnaryOp::Abs);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Abs, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())), Abs)
