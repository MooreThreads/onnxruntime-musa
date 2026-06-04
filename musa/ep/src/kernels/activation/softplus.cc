// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {
class Softplus : public OpKernelBase<Softplus> {
 public:
  Softplus(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Softplus::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape = info.GetShape();
  // muDNN SOFTPLUS returns incorrect values with the current MUSA 5.1.0 stack;
  // keep this op on the custom device fallback until the library path is fixed.
  return UnaryDeviceCompute(ctx, shape, elem_type, MusaUnaryOp::Softplus,
                            "Softplus");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Softplus, kOnnxDomain, 1, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    Softplus)
