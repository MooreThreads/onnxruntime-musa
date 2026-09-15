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

#include "shared_inc/op_kernel_common.h"

namespace {
class Erf : public OpKernelBase<Erf> {
 public:
  Erf(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Erf::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape = info.GetShape();
  if (OutputEmptyTensorIfNeeded(ctx, shape)) {
    return nullptr;
  }
  // muDNN ERF on Toolkit 4.3.8 returns SUCCESS but maps +/-Inf to NaN.
  // Use the existing device erff/erf implementation (as in CUDA/musify) for
  // all registered floating types to preserve infinities, NaNs and signed zero.
  return UnaryDeviceCompute(ctx, shape, elem_type, MusaUnaryOp::Erf, "Erf");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Erf, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    Erf)
