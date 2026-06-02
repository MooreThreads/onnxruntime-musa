// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Pad : public OpKernelBase<Pad> {
 public:
  Pad(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo ki(info);
    mode_ = AttrOrDefault<std::string>(ki, "mode", std::string("constant"));
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  std::string mode_;
};

OrtStatus* Pad::Compute(Ort::KernelContext& ctx) const {
  if (mode_ != "constant")
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Pad: only mode='constant' is supported");

  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();
  size_t rank = shape0.size();
  size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Pad: unsupported dtype");

  // pads: [begin_dim0, begin_dim1, ..., end_dim0, end_dim1, ...]
  std::vector<int64_t> pads = ReadIntTensor(ctx, 1);
  if (pads.size() != 2 * rank)
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Pad: pads size mismatch");

  // Optional constant_value (input 2)
  std::vector<uint8_t> fill_bytes(elem_size, 0);
  if (ctx.GetInputCount() >= 3) {
    std::vector<uint8_t> cv_bytes;
    RETURN_IF_ERROR(CopyToHost(ctx.GetInput(2), cv_bytes));
    if (!cv_bytes.empty()) fill_bytes = cv_bytes;
  }

  std::vector<int64_t> out_shape(rank);
  for (size_t i = 0; i < rank; ++i)
    out_shape[i] = shape0[i] + pads[i] + pads[i + rank];

  std::vector<uint8_t> in_bytes;
  RETURN_IF_ERROR(CopyToHost(input0, in_bytes));

  int64_t out_total = NumElements(out_shape);
  std::vector<uint8_t> out_bytes(static_cast<size_t>(out_total) * elem_size);
  // Fill with constant value
  for (int64_t i = 0; i < out_total; ++i)
    std::memcpy(out_bytes.data() + static_cast<size_t>(i) * elem_size,
                fill_bytes.data(), elem_size);

  // Copy input elements to the padded region
  auto in_strides = Strides(shape0);
  auto out_strides = Strides(out_shape);
  int64_t in_total = NumElements(shape0);
  for (int64_t i = 0; i < in_total; ++i) {
    auto ic = Coordinates(i, shape0);
    std::vector<int64_t> oc(rank);
    for (size_t d = 0; d < rank; ++d) oc[d] = ic[d] + pads[d];
    int64_t o_off = Offset(oc, out_strides);
    std::memcpy(out_bytes.data() + static_cast<size_t>(o_off) * elem_size,
                in_bytes.data() + static_cast<size_t>(i) * elem_size,
                elem_size);
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return CopyFromHost(y, out_bytes.data(), out_bytes.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pad, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", TensorTypesWithBool())),
    Pad)
