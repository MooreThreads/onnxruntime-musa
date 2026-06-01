// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Erf : public OpKernelBase<Erf> {
 public:
  Erf(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Erf::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Erf only supports float tensors");
  }
  return UnaryCompute<float>(ctx, info.GetShape(),
                             [](float x) { return std::erf(x); },
                             MusaUnaryOp::Erf);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Erf, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    Erf)
