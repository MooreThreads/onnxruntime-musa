// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/expand_impl.h"

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
    target_shape.insert(target_shape.begin(),
                        input_shape.size() - target_shape.size(), 1);
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
  if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Expand requires MUSA input");
  }
  if (!CanUseBroadcastKernel(out_shape, input_shape, out_shape)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Expand rank exceeds MUSA kernel limit");
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Expand requires MUSA output");
  }
  return LaunchStatus(LaunchMusaExpandKernel(
      input.GetTensorRawData(), y.GetTensorMutableRawData(),
      static_cast<int32_t>(elem_size),
      MakeBroadcastParams(out_shape, input_shape, out_shape), nullptr));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Expand, kOnnxDomain, 13, 19,
                                  (Ort::KernelDefBuilder()
                                       .AddTypeConstraint("T",
                                                          AllFixedSizeTensorTypes())),
                                  Expand)
