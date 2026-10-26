// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "shared_inc/blas_utils.h"

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
  return GemmCompute(ctx, trans_a_ != 0, trans_b_ != 0, alpha_, beta_,
                     std::string{}, 0.0f);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    Gemm)

// The Gemm version differences below opset 13 are type-set and optional-input
// changes, not arithmetic ones: opset 7 dropped the `broadcast` attribute,
// opset 9 added the integer element types (which this kernel does not
// implement, so only the floating-point types are declared) and opset 11 made
// the C input optional (the kernel already treats a missing third input as no
// bias). The ranges follow the CUDA EP's splits; the schemas have no bfloat16
// before opset 13.
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm, kOnnxDomain, 7, 8,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())), Gemm)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm, kOnnxDomain, 9, 10,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())), Gemm)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm, kOnnxDomain, 11, 12,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())), Gemm)
