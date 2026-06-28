// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {
class LessOrEqual : public OpKernelBase<LessOrEqual> {
 public:
  LessOrEqual(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* LessOrEqual::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape0 = info.GetShape();
  auto shape1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
  return CompareDeviceCompute(ctx, shape0, shape1, elem_type,
                              MusaCompareOp::LessOrEqual, "LessOrEqual");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(LessOrEqual, kOnnxDomain, 12, 15,
                                  (Ort::KernelDefBuilder().AddTypeConstraint(
                                      "T", CompareTensorTypesNoBFloat16())),
                                  LessOrEqual)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    LessOrEqual, kOnnxDomain, 16, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", CompareTensorTypes())),
    LessOrEqual)
