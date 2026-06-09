// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "logical/logical_ops_impl.h"
#include "shared_inc/op_kernel_common.h"

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
  Ort::ConstValue input_value = ctx.GetInput(0);
  auto shape = info.GetShape();
  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  if (IsGpuMemory(input_value.GetTensorMemoryInfo()) &&
      IsGpuMemory(y.GetTensorMemoryInfo())) {
    return LaunchStatus(LaunchMusaNotBoolKernel(
        input_value.GetTensorData<uint8_t>(), y.GetTensorMutableData<uint8_t>(),
        NumElements(shape), GetComputeStream(ctx)));
  }
  return UnsupportedDeviceElementwiseStatus("Not",
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Not, kOnnxDomain, 1, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint(
        "T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))),
    Not)
