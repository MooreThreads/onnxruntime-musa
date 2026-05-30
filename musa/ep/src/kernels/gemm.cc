// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/blas_utils.h"

namespace {
class Gemm : public OpKernelBase<Gemm> {
 public:
  Gemm(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    trans_a_ = AttrOrDefault<int64_t>(kernel_info, "transA", 0);
    trans_b_ = AttrOrDefault<int64_t>(kernel_info, "transB", 0);
    alpha_ = AttrOrDefault<float>(kernel_info, "alpha", 1.0f);
    beta_ = AttrOrDefault<float>(kernel_info, "beta", 1.0f);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t trans_a_ = 0;
  int64_t trans_b_ = 0;
  float alpha_ = 1.0f;
  float beta_ = 1.0f;
};

OrtStatus* Gemm::Compute(Ort::KernelContext& ctx) const {
  if (ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetElementType() !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Gemm only supports float tensors");
  }
  return GemmCompute(ctx, trans_a_ != 0, trans_b_ != 0, alpha_, beta_,
                     std::string{}, 0.0f);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())), Gemm)
