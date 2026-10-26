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
class Sin : public OpKernelBase<Sin> {
 public:
  Sin(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Sin::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape = info.GetShape();
  if (OutputEmptyTensorIfNeeded(ctx, shape)) {
    return nullptr;
  }
  return UnaryDeviceCompute(ctx, shape, elem_type, MusaUnaryOp::Sin, "Sin");
}
}  // namespace

// The Sin/Cos schemas are unchanged from opset 7 through 19: only the element
// types grow at opset 13 (bfloat16), so one registration covers both the old
// and the current models.
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sin, kOnnxDomain, 7, 12,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())), Sin)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sin, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    Sin)
