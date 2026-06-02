// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Where : public OpKernelBase<Where> {
 public:
  Where(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Where::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue cond_val = ctx.GetInput(0);
  Ort::ConstValue x_val = ctx.GetInput(1);
  Ort::ConstValue y_val = ctx.GetInput(2);

  auto cond_shape = cond_val.GetTensorTypeAndShapeInfo().GetShape();
  auto x_info = x_val.GetTensorTypeAndShapeInfo();
  auto x_shape = x_info.GetShape();
  auto y_shape = y_val.GetTensorTypeAndShapeInfo().GetShape();
  auto elem_type = x_info.GetElementType();
  size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Where: unsupported dtype");

  // Broadcast all three shapes
  std::vector<int64_t> out_shape =
      BroadcastShape(BroadcastShape(cond_shape, x_shape), y_shape);
  int64_t total = NumElements(out_shape);

  std::vector<uint8_t> cond_bytes, x_bytes, y_bytes;
  RETURN_IF_ERROR(CopyToHost(cond_val, cond_bytes));
  RETURN_IF_ERROR(CopyToHost(x_val, x_bytes));
  RETURN_IF_ERROR(CopyToHost(y_val, y_bytes));

  auto sc = Strides(cond_shape);
  auto sx = Strides(x_shape);
  auto sy = Strides(y_shape);

  std::vector<uint8_t> out_bytes(static_cast<size_t>(total) * elem_size);
  for (int64_t i = 0; i < total; ++i) {
    auto coord = Coordinates(i, out_shape);
    int64_t ci = BroadcastOffset(coord, cond_shape, sc);
    int64_t xi = BroadcastOffset(coord, x_shape, sx);
    int64_t yi = BroadcastOffset(coord, y_shape, sy);
    bool cond = cond_bytes[static_cast<size_t>(ci)] != 0;
    const uint8_t* src =
        cond ? x_bytes.data() + static_cast<size_t>(xi) * elem_size
             : y_bytes.data() + static_cast<size_t>(yi) * elem_size;
    std::memcpy(out_bytes.data() + static_cast<size_t>(i) * elem_size, src,
                elem_size);
  }

  Ort::UnownedValue out = ctx.GetOutput(0, out_shape);
  return CopyFromHost(out, out_bytes.data(), out_bytes.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Where, kOnnxDomain, 9, 17,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", TensorTypesWithBool())
         .AddTypeConstraint("B", BoolTensorTypes())),
    Where)
