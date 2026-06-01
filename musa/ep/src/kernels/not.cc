// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Not : public OpKernelBase<Not> {
 public:
  Not(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Not::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Not only supports bool tensors");
  }
  std::vector<uint8_t> input = ReadTyped<uint8_t>(ctx.GetInput(0));
  std::vector<uint8_t> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = static_cast<uint8_t>(!input[i]);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, info.GetShape());
  return CopyFromHost(y, output.data(), output.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Not, kOnnxDomain, 1, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint(
        "T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))),
    Not)
