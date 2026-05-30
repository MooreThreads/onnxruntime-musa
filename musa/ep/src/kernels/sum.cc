// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Sum : public OpKernelBase<Sum> {
 public:
  Sum(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Sum::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto info = input0.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Sum only supports float");
  }
  std::vector<int64_t> shape0 = info.GetShape();
  std::vector<int64_t> out_shape = shape0;
  std::vector<float> out = ReadTyped<float>(input0);
  for (size_t idx = 1; idx < ctx.GetInputCount(); ++idx) {
    auto shape = ctx.GetInput(idx).GetTensorTypeAndShapeInfo().GetShape();
    out_shape = BroadcastShape(out_shape, shape);
    std::vector<float> lhs = out;
    std::vector<int64_t> lhs_shape = idx == 1 ? shape0 : out_shape;
    std::vector<float> rhs = ReadTyped<float>(ctx.GetInput(idx));
    std::vector<float> next(static_cast<size_t>(NumElements(out_shape)), 0.0f);
    auto ls = Strides(lhs_shape), rs = Strides(shape);
    for (int64_t i = 0; i < NumElements(out_shape); ++i) {
      auto coord = Coordinates(i, out_shape);
      next[static_cast<size_t>(i)] =
          lhs[static_cast<size_t>(BroadcastOffset(coord, lhs_shape, ls))] +
          rhs[static_cast<size_t>(BroadcastOffset(coord, shape, rs))];
    }
    out = std::move(next);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return WriteTyped<float>(y, out);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sum, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), Sum)
