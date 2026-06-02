// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Tile : public OpKernelBase<Tile> {
 public:
  Tile(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Tile::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto info = input.GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape = info.GetShape();
  size_t rank = shape.size();
  size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Tile: unsupported dtype");

  std::vector<int64_t> repeats = ReadIntTensor(ctx, 1);
  if (repeats.size() != rank)
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Tile: repeats rank mismatch");

  std::vector<int64_t> out_shape(rank);
  for (size_t d = 0; d < rank; ++d) out_shape[d] = shape[d] * repeats[d];

  std::vector<uint8_t> in_bytes;
  RETURN_IF_ERROR(CopyToHost(input, in_bytes));

  auto in_strides = Strides(shape);
  int64_t out_total = NumElements(out_shape);
  std::vector<uint8_t> out_bytes(static_cast<size_t>(out_total) * elem_size);

  for (int64_t i = 0; i < out_total; ++i) {
    auto oc = Coordinates(i, out_shape);
    std::vector<int64_t> ic(rank);
    for (size_t d = 0; d < rank; ++d) ic[d] = oc[d] % shape[d];
    int64_t in_off = Offset(ic, in_strides);
    std::memcpy(out_bytes.data() + static_cast<size_t>(i) * elem_size,
                in_bytes.data() + static_cast<size_t>(in_off) * elem_size,
                elem_size);
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return CopyFromHost(y, out_bytes.data(), out_bytes.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Tile, kOnnxDomain, 6, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", TensorTypesWithBool())),
    Tile)
