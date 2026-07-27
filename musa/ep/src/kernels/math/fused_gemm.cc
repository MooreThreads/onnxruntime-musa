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
class FusedGemm : public OpKernelBase<FusedGemm> {
 public:
  FusedGemm(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    trans_a_ = AttrOrDefault<int64_t>(kernel_info, "transA", 0);
    trans_b_ = AttrOrDefault<int64_t>(kernel_info, "transB", 0);
    alpha_ = AttrOrDefault<float>(kernel_info, "alpha", 1.0f);
    beta_ = AttrOrDefault<float>(kernel_info, "beta", 1.0f);
    activation_ = AttrOrDefault<std::string>(kernel_info, "activation", "");
    activation_alpha_ =
        AttrOrDefault<float>(kernel_info, "activation_alpha", 0.01f);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t trans_a_ = 0;
  int64_t trans_b_ = 0;
  float alpha_ = 1.0f;
  float beta_ = 1.0f;
  std::string activation_;
  float activation_alpha_ = 0.01f;
};

OrtStatus* FusedGemm::Compute(Ort::KernelContext& ctx) const {
  return GemmCompute(ctx, trans_a_ != 0, trans_b_ != 0, alpha_, beta_,
                     activation_, activation_alpha_);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    FusedGemm, kMSDomain, 1, 1,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    FusedGemm)
