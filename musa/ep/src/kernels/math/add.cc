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
#include "shared_inc/op_kernel_common.h"

namespace {
bool TryMudnnAdd(Ort::KernelContext& ctx, const std::vector<int64_t>& shape0,
                 const std::vector<int64_t>& shape1,
                 ONNXTensorElementDataType elem_type) {
  return TryMudnnBinary(ctx, shape0, shape1, elem_type,
                        ::musa::dnn::Binary::Mode::ADD);
}

class Add : public OpKernelBase<Add> {
 public:
  Add(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Add::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape0 = info.GetShape();
  auto shape1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
  auto out_shape = BroadcastShape(shape0, shape1);
  if (NumElements(out_shape) == 0) {
    ctx.GetOutput(0, out_shape);
    return nullptr;
  }
  if (TryMudnnAdd(ctx, shape0, shape1, elem_type)) {
    return nullptr;
  }
  return BinaryDeviceCompute(ctx, shape0, shape1, elem_type, MusaBinaryOp::Add,
                             "Add");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Add, kOnnxDomain, 13, 13,
                                  (Ort::KernelDefBuilder().AddTypeConstraint(
                                      "T", BinaryNumericOpset13TensorTypes())),
                                  Add)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Add, kOnnxDomain, 14, 19,
                                  (Ort::KernelDefBuilder().AddTypeConstraint(
                                      "T", BinaryNumericOpset14TensorTypes())),
                                  Add)
