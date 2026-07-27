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

#include "logical/logical_ops_impl.h"
#include "shared_inc/op_kernel_common.h"

namespace {
class Or : public OpKernelBase<Or> {
 public:
  Or(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Or::Compute(Ort::KernelContext& ctx) const {
  auto info0 = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto info1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo();
  if (info0.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL ||
      info1.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Or only supports bool tensors");
  }
  auto shape0 = info0.GetShape();
  auto shape1 = info1.GetShape();
  auto out_shape = BroadcastShape(shape0, shape1);
  Ort::ConstValue lhs_value = ctx.GetInput(0);
  Ort::ConstValue rhs_value = ctx.GetInput(1);
  if (IsGpuMemory(lhs_value.GetTensorMemoryInfo()) &&
      IsGpuMemory(rhs_value.GetTensorMemoryInfo()) &&
      CanUseBroadcastKernel(out_shape, shape0, shape1)) {
    Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
    if (IsGpuMemory(y.GetTensorMemoryInfo())) {
      MusaBroadcastParams params =
          MakeBroadcastParams(out_shape, shape0, shape1);
      return LaunchStatus(LaunchMusaOrBoolKernel(
          lhs_value.GetTensorData<uint8_t>(),
          rhs_value.GetTensorData<uint8_t>(), y.GetTensorMutableData<uint8_t>(),
          params, GetComputeStream(ctx)));
    }
  }
  return UnsupportedDeviceElementwiseStatus("Or",
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Or, kOnnxDomain, 7, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint(
        "T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))),
    Or)
