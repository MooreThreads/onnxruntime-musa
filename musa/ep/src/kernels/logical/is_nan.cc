// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {
class IsNaN : public OpKernelBase<IsNaN> {
 public:
  IsNaN(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* IsNaN::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto info = input.GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape = info.GetShape();

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      !IsGpuMemory(input.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus("IsNaN", elem_type);
  }

  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus("IsNaN", elem_type);
  }

  musaError_t status =
      LaunchMusaIsNaNKernel(input.GetTensorRawData(),
                            y.GetTensorMutableData<uint8_t>(),
                            NumElements(shape), musa_elem_type, nullptr);
  if (status == musaErrorNotSupported) {
    return UnsupportedDeviceElementwiseStatus("IsNaN", elem_type);
  }
  return LaunchStatus(status);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    IsNaN, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T1", FloatBoolTensorTypes())
         .AddTypeConstraint(
             "T2", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))),
    IsNaN)
