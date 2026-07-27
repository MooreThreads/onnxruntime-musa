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
class Not : public OpKernelBase<Not> {
 public:
  Not(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Not::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Not only supports bool tensors");
  }
  Ort::ConstValue input_value = ctx.GetInput(0);
  auto shape = info.GetShape();
  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  if (IsGpuMemory(input_value.GetTensorMemoryInfo()) &&
      IsGpuMemory(y.GetTensorMemoryInfo())) {
    return LaunchStatus(LaunchMusaNotBoolKernel(
        input_value.GetTensorData<uint8_t>(), y.GetTensorMutableData<uint8_t>(),
        NumElements(shape), GetComputeStream(ctx)));
  }
  return UnsupportedDeviceElementwiseStatus("Not",
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Not, kOnnxDomain, 1, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint(
        "T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))),
    Not)
