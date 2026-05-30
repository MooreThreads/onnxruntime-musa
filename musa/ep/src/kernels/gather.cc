// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Gather : public OpKernelBase<Gather> {
 public:
  Gather(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = 0;
};

OrtStatus* Gather::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();
  int64_t axis = NormalizeAxis(axis_, shape0.size());
  std::vector<int64_t> indices = ReadIntTensor(ctx, 1);
  auto indices_shape = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> out_shape;
  out_shape.insert(out_shape.end(), shape0.begin(), shape0.begin() + axis);
  out_shape.insert(out_shape.end(), indices_shape.begin(), indices_shape.end());
  out_shape.insert(out_shape.end(), shape0.begin() + axis + 1, shape0.end());
  size_t elem_size = ElementSize(elem_type);
  std::vector<uint8_t> in;
  RETURN_IF_ERROR(CopyToHost(input0, in));
  std::vector<uint8_t> out(static_cast<size_t>(NumElements(out_shape)) *
                           elem_size);
  auto in_strides = Strides(shape0);
  for (int64_t i = 0; i < NumElements(out_shape); ++i) {
    auto oc = Coordinates(i, out_shape);
    std::vector<int64_t> ic(shape0.size(), 0);
    for (int64_t d = 0; d < axis; ++d)
      ic[static_cast<size_t>(d)] = oc[static_cast<size_t>(d)];
    int64_t idx_offset = 0;
    auto idx_strides = Strides(indices_shape);
    for (size_t j = 0; j < indices_shape.size(); ++j)
      idx_offset += oc[static_cast<size_t>(axis) + j] * idx_strides[j];
    int64_t gather_idx = indices[static_cast<size_t>(idx_offset)];
    if (gather_idx < 0) gather_idx += shape0[static_cast<size_t>(axis)];
    ic[static_cast<size_t>(axis)] = gather_idx;
    for (size_t d = static_cast<size_t>(axis) + 1; d < shape0.size(); ++d) {
      ic[d] = oc[d - 1 + indices_shape.size()];
    }
    std::memcpy(
        out.data() + static_cast<size_t>(i) * elem_size,
        in.data() + static_cast<size_t>(Offset(ic, in_strides)) * elem_size,
        elem_size);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return CopyFromHost(y, out.data(), out.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Gather, kOnnxDomain, 13, 17,
                                  (Ort::KernelDefBuilder()
                                       .AddTypeConstraint("T", AllTensorTypes())
                                       .AddTypeConstraint("Tind",
                                                          IntTensorTypes())),
                                  Gather)
