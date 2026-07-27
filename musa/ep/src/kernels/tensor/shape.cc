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
class Shape : public OpKernelBase<Shape> {
 public:
  Shape(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Shape::Compute(Ort::KernelContext& ctx) const {
  auto shape0 = ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> out(shape0.begin(), shape0.end());
  Ort::UnownedValue y = ctx.GetOutput(0, {static_cast<int64_t>(out.size())});
  return WriteTyped<int64_t>(y, out, GetComputeStream(ctx));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .AddTypeConstraint("T1",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .SetOutputMemType(0, OrtMemTypeCPUInput)),
    Shape)
