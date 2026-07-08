// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {
class Neg : public OpKernelBase<Neg> {
 public:
  Neg(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Neg::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  return UnaryDeviceCompute(ctx, info.GetShape(), info.GetElementType(),
                            MusaUnaryOp::Neg, "Neg");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Neg, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", NegTensorTypes())), Neg)
