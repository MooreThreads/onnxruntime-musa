// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {
class Flatten : public OpKernelBase<Flatten> {
 public:
  Flatten(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", 1);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = 1;
};

OrtStatus* Flatten::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto info = input.GetTensorTypeAndShapeInfo();
  auto input_shape = info.GetShape();
  int64_t axis = axis_;
  if (axis < 0) {
    axis += static_cast<int64_t>(input_shape.size());
  }
  if (axis < 0 || axis > static_cast<int64_t>(input_shape.size())) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Flatten axis out of range");
  }

  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) {
    outer *= input_shape[static_cast<size_t>(i)];
  }
  int64_t inner = 1;
  for (size_t i = static_cast<size_t>(axis); i < input_shape.size(); ++i) {
    inner *= input_shape[i];
  }
  Ort::UnownedValue output = ctx.GetOutput(0, {outer, inner});
  if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Flatten requires MUSA device tensors");
  }
  return CopyRawTensor(input, output, input.GetTensorSizeInBytes());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Flatten, kOnnxDomain, 1, 8,
                                  (Ort::KernelDefBuilder()
                                       .AddTypeConstraint("T", HfdTensorTypes())
                                       .AddInputOutputAlias(0, 0)),
                                  Flatten)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten, kOnnxDomain, 9, 10,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypesNoBFloat16())
         .AddInputOutputAlias(0, 0)),
    Flatten)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten, kOnnxDomain, 11, 12,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypesNoBFloat16())
         .AddInputOutputAlias(0, 0)),
    Flatten)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .AddInputOutputAlias(0, 0)),
    Flatten)
