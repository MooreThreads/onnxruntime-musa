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
class Unsqueeze : public OpKernelBase<Unsqueeze> {
 public:
  Unsqueeze(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axes_attr_ = AttrsOrEmpty(kernel_info, "axes");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  std::vector<int64_t> axes_attr_;
};

OrtStatus* Unsqueeze::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto shape0 = input0.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> axes = axes_attr_;
  if (ctx.GetInputCount() > 1) axes = ReadIntTensor(ctx, 1);
  std::vector<int64_t> out_shape;
  int64_t out_rank = static_cast<int64_t>(shape0.size() + axes.size());
  std::set<int64_t> ax;
  for (int64_t a : axes) ax.insert(a < 0 ? a + out_rank : a);
  size_t src = 0;
  for (int64_t i = 0; i < out_rank; ++i) {
    out_shape.push_back(ax.count(i) ? 1 : shape0[src++]);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return CopyRawTensor(input0, y, input0.GetTensorSizeInBytes(),
                       GetComputeStream(ctx));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .SetInputMemType(1, OrtMemTypeCPUInput)
         .AddInputOutputAlias(0, 0)),
    Unsqueeze)
