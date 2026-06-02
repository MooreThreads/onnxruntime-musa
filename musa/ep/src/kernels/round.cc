// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Round : public OpKernelBase<Round> {
 public:
  Round(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Round::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Round: only float32 supported");
  // ONNX Round uses "round half to even" (banker's rounding)
  return UnaryCompute<float>(ctx, info.GetShape(), [](float x) {
    return std::rint(x);  // std::rint respects current rounding mode (default =
                          // nearest-even)
  });
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Round, kOnnxDomain, 11, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())), Round)
