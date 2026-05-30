// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Split : public OpKernelBase<Split> {
 public:
  Split(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = 0;
};

OrtStatus* Split::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();
  int64_t axis = NormalizeAxis(axis_, shape0.size());
  std::vector<int64_t> splits;
  if (ctx.GetInputCount() > 1) {
    splits = ReadIntTensor(ctx, 1);
  } else {
    size_t count = ctx.GetOutputCount();
    splits.assign(
        count, shape0[static_cast<size_t>(axis)] / static_cast<int64_t>(count));
  }
  size_t elem_size = ElementSize(elem_type);
  std::vector<uint8_t> in;
  RETURN_IF_ERROR(CopyToHost(input0, in));
  auto in_strides = Strides(shape0);
  int64_t axis_start = 0;
  for (size_t out_idx = 0; out_idx < splits.size(); ++out_idx) {
    std::vector<int64_t> out_shape = shape0;
    out_shape[static_cast<size_t>(axis)] = splits[out_idx];
    std::vector<uint8_t> out(static_cast<size_t>(NumElements(out_shape)) *
                             elem_size);
    for (int64_t i = 0; i < NumElements(out_shape); ++i) {
      auto oc = Coordinates(i, out_shape);
      auto ic = oc;
      ic[static_cast<size_t>(axis)] += axis_start;
      std::memcpy(
          out.data() + static_cast<size_t>(i) * elem_size,
          in.data() + static_cast<size_t>(Offset(ic, in_strides)) * elem_size,
          elem_size);
    }
    Ort::UnownedValue y = ctx.GetOutput(out_idx, out_shape);
    RETURN_IF_ERROR(CopyFromHost(y, out.data(), out.size()));
    axis_start += splits[out_idx];
  }
  return nullptr;
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Split, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), Split)
