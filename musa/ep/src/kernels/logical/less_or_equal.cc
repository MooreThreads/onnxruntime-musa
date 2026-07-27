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
class LessOrEqual : public OpKernelBase<LessOrEqual> {
 public:
  LessOrEqual(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* LessOrEqual::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape0 = info.GetShape();
  auto shape1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
  return CompareDeviceCompute(ctx, shape0, shape1, elem_type,
                              MusaCompareOp::LessOrEqual, "LessOrEqual");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(LessOrEqual, kOnnxDomain, 12, 15,
                                  (Ort::KernelDefBuilder().AddTypeConstraint(
                                      "T", CompareTensorTypesNoBFloat16())),
                                  LessOrEqual)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    LessOrEqual, kOnnxDomain, 16, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", CompareTensorTypes())),
    LessOrEqual)
