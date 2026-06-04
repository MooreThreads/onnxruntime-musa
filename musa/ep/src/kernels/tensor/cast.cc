// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/cast_op_impl.h"

namespace {
class Cast : public OpKernelBase<Cast> {
 public:
  Cast(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    to_ = AttrOrDefault<int64_t>(kernel_info, "to", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t to_ = 0;
};

OrtStatus* Cast::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();
  const size_t src_elem_size = ElementSize(elem_type);
  const size_t dst_elem_size =
      ElementSize(static_cast<ONNXTensorElementDataType>(to_));
  if (src_elem_size == 0 || dst_elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Cast unsupported dtype");
  }

  Ort::UnownedValue y = ctx.GetOutput(0, shape0);
  if (!IsGpuMemory(input0.GetTensorMemoryInfo()) ||
      !IsGpuMemory(y.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Cast requires MUSA input and output");
  }

  const int64_t n = NumElements(shape0);
  if (static_cast<int64_t>(elem_type) == to_) {
    return DeviceMemcpy(y.GetTensorMutableRawData(), input0.GetTensorRawData(),
                        static_cast<size_t>(n) * src_elem_size);
  }

  return LaunchStatus(LaunchMusaCastKernel(
      input0.GetTensorRawData(), y.GetTensorMutableRawData(),
      static_cast<int32_t>(elem_type), static_cast<int32_t>(to_), n, nullptr));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T1", TensorTypesWithBool())
         .AddTypeConstraint("T2", TensorTypesWithBool())),
    Cast)
