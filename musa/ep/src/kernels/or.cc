// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Or : public OpKernelBase<Or> {
 public:
  Or(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Or::Compute(Ort::KernelContext& ctx) const {
  auto info0 = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto info1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo();
  if (info0.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL ||
      info1.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Or only supports bool tensors");
  }
  auto shape0 = info0.GetShape();
  auto shape1 = info1.GetShape();
  auto out_shape = BroadcastShape(shape0, shape1);
  std::vector<uint8_t> lhs = ReadTyped<uint8_t>(ctx.GetInput(0));
  std::vector<uint8_t> rhs = ReadTyped<uint8_t>(ctx.GetInput(1));
  std::vector<uint8_t> output(static_cast<size_t>(NumElements(out_shape)));
  auto s0 = Strides(shape0);
  auto s1 = Strides(shape1);
  for (int64_t i = 0; i < NumElements(out_shape); ++i) {
    auto coord = Coordinates(i, out_shape);
    output[static_cast<size_t>(i)] = static_cast<uint8_t>(
        lhs[static_cast<size_t>(BroadcastOffset(coord, shape0, s0))] ||
        rhs[static_cast<size_t>(BroadcastOffset(coord, shape1, s1))]);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return CopyFromHost(y, output.data(), output.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Or, kOnnxDomain, 7, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint(
        "T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))),
    Or)
