// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Softplus : public OpKernelBase<Softplus> {
 public:
  Softplus(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Softplus::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Softplus: only float32 supported");
  // softplus(x) = ln(1 + e^x)
  return UnaryCompute<float>(ctx, info.GetShape(),
                             [](float x) { return std::log1p(std::exp(x)); });
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Softplus, kOnnxDomain, 1, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    Softplus)
