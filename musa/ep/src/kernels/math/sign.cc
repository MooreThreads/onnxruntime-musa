// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {
class Sign : public OpKernelBase<Sign> {
 public:
  Sign(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Sign::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  return UnaryDeviceCompute(ctx, info.GetShape(), info.GetElementType(),
                            MusaUnaryOp::Sign, "Sign");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sign, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", SignTensorTypes())), Sign)
