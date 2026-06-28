// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {
class Floor : public OpKernelBase<Floor> {
 public:
  Floor(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Floor::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape = info.GetShape();
  return UnaryDeviceCompute(ctx, shape, elem_type, MusaUnaryOp::Floor, "Floor");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Floor, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())), Floor)
