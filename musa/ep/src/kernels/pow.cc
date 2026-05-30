// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Pow : public OpKernelBase<Pow> {
 public:
  Pow(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Pow::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape0 = info.GetShape();
  auto shape1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return BinaryCompute<float>(
        ctx, shape0, shape1, [](float a, float b) { return std::pow(a, b); });
  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                    "unsupported binary op dtype");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pow, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), Pow)
