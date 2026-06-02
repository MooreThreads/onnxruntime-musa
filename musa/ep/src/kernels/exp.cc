// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Exp : public OpKernelBase<Exp> {
 public:
  Exp(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Exp::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Exp: only float32 supported");
  return UnaryCompute<float>(ctx, info.GetShape(),
                             [](float x) { return std::exp(x); });
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Exp, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())), Exp)
