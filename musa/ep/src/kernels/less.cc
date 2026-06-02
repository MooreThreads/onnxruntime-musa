// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
template <typename T>
OrtStatus* LessTyped(Ort::KernelContext& ctx,
                     const std::vector<int64_t>& shape0,
                     const std::vector<int64_t>& shape1) {
  std::vector<T> a = ReadTyped<T>(ctx.GetInput(0));
  std::vector<T> b = ReadTyped<T>(ctx.GetInput(1));
  std::vector<int64_t> out_shape = BroadcastShape(shape0, shape1);
  std::vector<uint8_t> out(static_cast<size_t>(NumElements(out_shape)));
  auto s0 = Strides(shape0);
  auto s1 = Strides(shape1);
  for (int64_t i = 0; i < NumElements(out_shape); ++i) {
    auto coord = Coordinates(i, out_shape);
    out[static_cast<size_t>(i)] = static_cast<uint8_t>(
        a[static_cast<size_t>(BroadcastOffset(coord, shape0, s0))] <
        b[static_cast<size_t>(BroadcastOffset(coord, shape1, s1))]);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return CopyFromHost(y, out.data(), out.size());
}

class Less : public OpKernelBase<Less> {
 public:
  Less(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Less::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape0 = info.GetShape();
  auto shape1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return LessTyped<float>(ctx, shape0, shape1);
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
    return LessTyped<int32_t>(ctx, shape0, shape1);
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    return LessTyped<int64_t>(ctx, shape0, shape1);
  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                    "Less: unsupported dtype");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Less, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), Less)
