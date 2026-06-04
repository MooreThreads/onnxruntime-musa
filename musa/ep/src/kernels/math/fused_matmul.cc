// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "math/matmul.h"
#include "shared_inc/blas_utils.h"

namespace {
class FusedMatMul : public OpKernelBase<FusedMatMul> {
 public:
  FusedMatMul(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    trans_a_ = AttrOrDefault<int64_t>(kernel_info, "transA", 0);
    trans_b_ = AttrOrDefault<int64_t>(kernel_info, "transB", 0);
    trans_batch_a_ = AttrOrDefault<int64_t>(kernel_info, "transBatchA", 0);
    trans_batch_b_ = AttrOrDefault<int64_t>(kernel_info, "transBatchB", 0);
    alpha_ = AttrOrDefault<float>(kernel_info, "alpha", 1.0f);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t trans_a_ = 0;
  int64_t trans_b_ = 0;
  int64_t trans_batch_a_ = 0;
  int64_t trans_batch_b_ = 0;
  float alpha_ = 1.0f;
};

OrtStatus* FusedMatMul::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue a = ctx.GetInput(0);
  Ort::ConstValue b = ctx.GetInput(1);
  auto a_info = a.GetTensorTypeAndShapeInfo();
  auto b_info = b.GetTensorTypeAndShapeInfo();
  const auto elem_type = a_info.GetElementType();
  if (b_info.GetElementType() != elem_type) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "FusedMatMul input dtypes must match");
  }
  if (!IsGpuMemory(a.GetTensorMemoryInfo()) ||
      !IsGpuMemory(b.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "FusedMatMul requires MUSA inputs");
  }

  std::vector<int64_t> a_shape = a_info.GetShape();
  std::vector<int64_t> b_shape = b_info.GetShape();
  std::vector<int64_t> out_shape;
  RETURN_IF_ERROR(ComputeMusaMatMulOutputShape(
      a_shape, b_shape, trans_a_ != 0, trans_b_ != 0, trans_batch_a_ != 0,
      trans_batch_b_ != 0, out_shape));

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "FusedMatMul requires MUSA output");
  }

  return ComputeMusaMatMulDevice(
      a.GetTensorRawData(), b.GetTensorRawData(), y.GetTensorMutableRawData(),
      elem_type, a_shape, b_shape, out_shape, trans_a_ != 0, trans_b_ != 0,
      trans_batch_a_ != 0, trans_batch_b_ != 0, alpha_);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    FusedMatMul, kMSDomain, 1, 1,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    FusedMatMul)
