// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "logical/logical_ops_impl.h"
#include "shared_inc/op_kernel_common.h"

namespace {
class BitwiseAnd : public OpKernelBase<BitwiseAnd> {
 public:
  BitwiseAnd(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* BitwiseAnd::Compute(Ort::KernelContext& ctx) const {
  auto info0 = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto info1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo();
  auto elem_type = info0.GetElementType();
  if (elem_type != info1.GetElementType()) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "BitwiseAnd requires matching input dtypes");
  }
  auto shape0 = info0.GetShape();
  auto shape1 = info1.GetShape();
  auto out_shape = BroadcastShape(shape0, shape1);

  MusaElementType musa_elem_type;
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      !IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo()) ||
      !CanUseBroadcastKernel(out_shape, shape0, shape1)) {
    return UnsupportedDeviceElementwiseStatus("BitwiseAnd", elem_type);
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus("BitwiseAnd", elem_type);
  }

  musaError_t status =
      LaunchMusaBitwiseAndKernel(lhs.GetTensorRawData(), rhs.GetTensorRawData(),
                                 y.GetTensorMutableRawData(),
                                 MakeBroadcastParams(out_shape, shape0, shape1),
                                 musa_elem_type, GetComputeStream(ctx));
  if (status == musaErrorNotSupported) {
    return UnsupportedDeviceElementwiseStatus("BitwiseAnd", elem_type);
  }
  return LaunchStatus(status);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(BitwiseAnd, kOnnxDomain, 18, 19,
                                  (Ort::KernelDefBuilder().AddTypeConstraint(
                                      "T", BitwiseIntegerTensorTypes())),
                                  BitwiseAnd)
