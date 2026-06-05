// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {
class IdentityOp : public OpKernelBase<IdentityOp> {
 public:
  IdentityOp(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* IdentityOp::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto info = input.GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  if (ElementSize(elem_type) == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Identity unsupported dtype");
  }
  auto shape = info.GetShape();
  Ort::UnownedValue output = ctx.GetOutput(0, shape);
  if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Identity requires MUSA device tensors");
  }
  return CopyRawTensor(input, output, input.GetTensorSizeInBytes());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity, kOnnxDomain, 1, 12,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypesNoBFloat16())
         .AddInputOutputAlias(0, 0)),
    IdentityOp)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity, kOnnxDomain, 13, 13,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .AddInputOutputAlias(0, 0)),
    IdentityOp)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity, kOnnxDomain, 14, 18,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("V", AllFixedSizeTensorTypes())
         .AddInputOutputAlias(0, 0)),
    IdentityOp)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity, kOnnxDomain, 19, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("V", AllFixedSizeTensorTypes())
         .AddInputOutputAlias(0, 0)),
    IdentityOp)
