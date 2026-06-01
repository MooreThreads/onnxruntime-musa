// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Expand : public OpKernelBase<Expand> {
 public:
  Expand(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Expand::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto info = input.GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto input_shape = info.GetShape();
  std::vector<int64_t> target_shape = ReadIntTensor(ctx, 1);
  if (target_shape.size() < input_shape.size()) {
    target_shape.insert(target_shape.begin(), input_shape.size() - target_shape.size(), 1);
  }
  const size_t offset = target_shape.size() - input_shape.size();
  for (size_t i = 0; i < target_shape.size(); ++i) {
    if (target_shape[i] == -1 && i >= offset) {
      target_shape[i] = input_shape[i - offset];
    }
  }
  std::vector<int64_t> out_shape = BroadcastShape(input_shape, target_shape);
  const size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Expand unsupported dtype");
  }
  std::vector<uint8_t> input_bytes;
  RETURN_IF_ERROR(CopyToHost(input, input_bytes));
  std::vector<uint8_t> output(static_cast<size_t>(NumElements(out_shape)) * elem_size);
  auto input_strides = Strides(input_shape);
  for (int64_t i = 0; i < NumElements(out_shape); ++i) {
    auto coord = Coordinates(i, out_shape);
    int64_t input_offset = BroadcastOffset(coord, input_shape, input_strides);
    std::memcpy(output.data() + static_cast<size_t>(i) * elem_size,
                input_bytes.data() + static_cast<size_t>(input_offset) * elem_size,
                elem_size);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return CopyFromHost(y, output.data(), output.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Expand, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", TensorTypesWithBool())),
    Expand)
