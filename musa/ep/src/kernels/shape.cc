// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Shape : public OpKernelBase<Shape> {
 public:
  Shape(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Shape::Compute(Ort::KernelContext& ctx) const {
  auto shape0 = ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> out(shape0.begin(), shape0.end());
  Ort::UnownedValue y = ctx.GetOutput(0, {static_cast<int64_t>(out.size())});
  return WriteTyped<int64_t>(y, out);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder()
        .AddTypeConstraint("T", TensorTypesWithBool())
        .SetOutputMemType(0, OrtMemTypeCPUOutput)), Shape)
