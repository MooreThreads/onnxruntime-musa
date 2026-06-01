// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
template <typename T>
OrtStatus* MinComputeTyped(Ort::KernelContext& ctx) {
  std::vector<std::vector<int64_t>> shapes;
  std::vector<std::vector<T>> inputs;
  std::vector<int64_t> out_shape =
      ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetShape();
  for (size_t i = 0; i < ctx.GetInputCount(); ++i) {
    auto v = ctx.GetInput(i);
    shapes.push_back(v.GetTensorTypeAndShapeInfo().GetShape());
    out_shape = BroadcastShape(out_shape, shapes.back());
    inputs.push_back(ReadTyped<T>(v));
  }
  std::vector<T> out(static_cast<size_t>(NumElements(out_shape)));
  std::vector<std::vector<int64_t>> strides;
  for (const auto& s : shapes) strides.push_back(Strides(s));
  for (int64_t i = 0; i < NumElements(out_shape); ++i) {
    auto coord = Coordinates(i, out_shape);
    T value = inputs[0][static_cast<size_t>(
        BroadcastOffset(coord, shapes[0], strides[0]))];
    for (size_t input_idx = 1; input_idx < inputs.size(); ++input_idx) {
      T other = inputs[input_idx][static_cast<size_t>(
          BroadcastOffset(coord, shapes[input_idx], strides[input_idx]))];
      value = std::min(value, other);
    }
    out[static_cast<size_t>(i)] = value;
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return WriteTyped<T>(y, out);
}

class Min : public OpKernelBase<Min> {
 public:
  Min(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Min::Compute(Ort::KernelContext& ctx) const {
  auto elem_type = ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    if (ctx.GetInputCount() == 2) {
      auto shape0 = ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetShape();
      auto shape1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
      return BinaryCompute<float>(ctx, shape0, shape1,
                                  [](float a, float b) { return std::min(a, b); },
                                  MusaBinaryOp::Min);
    }
    return MinComputeTyped<float>(ctx);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
    return MinComputeTyped<int32_t>(ctx);
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    return MinComputeTyped<int64_t>(ctx);
  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, "Min unsupported dtype");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Min, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), Min)
