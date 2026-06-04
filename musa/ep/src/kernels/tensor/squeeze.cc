// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {
class Squeeze : public OpKernelBase<Squeeze> {
 public:
  Squeeze(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axes_attr_ = AttrsOrEmpty(kernel_info, "axes");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  std::vector<int64_t> axes_attr_;
};

OrtStatus* Squeeze::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto shape0 = input0.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> axes = axes_attr_;
  if (ctx.GetInputCount() > 1) axes = ReadIntTensor(ctx, 1);
  std::vector<int64_t> out_shape;
  std::set<int64_t> ax = AxesSet(axes, shape0.size());
  for (size_t i = 0; i < shape0.size(); ++i) {
    if (!ax.count(static_cast<int64_t>(i))) out_shape.push_back(shape0[i]);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return CopyRawTensor(input0, y, input0.GetTensorSizeInBytes());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .SetInputMemType(1, OrtMemTypeCPUInput)
         .AddInputOutputAlias(0, 0)), Squeeze)
