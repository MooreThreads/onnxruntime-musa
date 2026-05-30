// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Concat : public OpKernelBase<Concat> {
 public:
  Concat(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = 0;
};

OrtStatus* Concat::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();
  int64_t axis = NormalizeAxis(axis_, shape0.size());
  std::vector<std::vector<int64_t>> shapes;
  std::vector<std::vector<uint8_t>> inputs;
  size_t elem_size = ElementSize(elem_type);
  std::vector<int64_t> out_shape = shape0;
  out_shape[static_cast<size_t>(axis)] = 0;
  for (size_t i = 0; i < ctx.GetInputCount(); ++i) {
    auto v = ctx.GetInput(i);
    shapes.push_back(v.GetTensorTypeAndShapeInfo().GetShape());
    inputs.emplace_back();
    RETURN_IF_ERROR(CopyToHost(v, inputs.back()));
    out_shape[static_cast<size_t>(axis)] +=
        shapes.back()[static_cast<size_t>(axis)];
  }
  std::vector<uint8_t> out(static_cast<size_t>(NumElements(out_shape)) *
                           elem_size);
  auto out_strides = Strides(out_shape);
  std::vector<int64_t> axis_offsets;
  int64_t acc = 0;
  for (const auto& s : shapes) {
    axis_offsets.push_back(acc);
    acc += s[static_cast<size_t>(axis)];
  }
  for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
    int64_t total = NumElements(shapes[input_idx]);
    for (int64_t i = 0; i < total; ++i) {
      auto coord = Coordinates(i, shapes[input_idx]);
      auto out_coord = coord;
      out_coord[static_cast<size_t>(axis)] += axis_offsets[input_idx];
      std::memcpy(
          out.data() +
              static_cast<size_t>(Offset(out_coord, out_strides)) * elem_size,
          inputs[input_idx].data() + static_cast<size_t>(i) * elem_size,
          elem_size);
    }
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return CopyFromHost(y, out.data(), out.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Concat, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), Concat)
