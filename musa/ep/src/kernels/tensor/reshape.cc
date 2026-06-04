// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {
class Reshape : public OpKernelBase<Reshape> {
 public:
  Reshape(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    allowzero_ = AttrOrDefault<int64_t>(kernel_info, "allowzero", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t allowzero_ = 0;
};

OrtStatus* Reshape::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto shape0 = input0.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> out_shape = ReadIntTensor(ctx, 1);
  int64_t input_size = NumElements(shape0);
  int64_t known = 1;
  int infer_idx = -1;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    if (out_shape[i] == 0 && !allowzero_) out_shape[i] = shape0[i];
    if (out_shape[i] == -1) {
      infer_idx = static_cast<int>(i);
    } else {
      known *= out_shape[i];
    }
  }
  if (infer_idx >= 0)
    out_shape[static_cast<size_t>(infer_idx)] = input_size / known;
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return CopyRawTensor(input0, y, input0.GetTensorSizeInBytes());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .SetInputMemType(1, OrtMemTypeCPUInput)
         .AddInputOutputAlias(0, 0)), Reshape)
