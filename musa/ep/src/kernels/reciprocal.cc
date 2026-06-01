// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Reciprocal : public OpKernelBase<Reciprocal> {
 public:
  Reciprocal(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Reciprocal::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "unsupported unary op dtype");
  }
  return UnaryCompute<float>(ctx, info.GetShape(),
                             [](float x) { return 1.0f / x; },
                             MusaUnaryOp::Reciprocal);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reciprocal, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    Reciprocal)
