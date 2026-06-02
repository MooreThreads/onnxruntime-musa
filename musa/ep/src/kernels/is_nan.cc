// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class IsNaN : public OpKernelBase<IsNaN> {
 public:
  IsNaN(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* IsNaN::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "IsNaN: only float32 supported");
  auto shape = info.GetShape();
  std::vector<float> x = ReadTyped<float>(ctx.GetInput(0));
  std::vector<uint8_t> out(x.size());
  for (size_t i = 0; i < x.size(); ++i)
    out[i] = static_cast<uint8_t>(std::isnan(x[i]));
  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  return CopyFromHost(y, out.data(), out.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    IsNaN, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T1", FloatTensorTypes())),
    IsNaN)
