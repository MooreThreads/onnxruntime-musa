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
class IsNaN : public OpKernelBase<IsNaN> {
 public:
  IsNaN(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* IsNaN::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto info = input.GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape = info.GetShape();

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      !IsGpuMemory(input.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus("IsNaN", elem_type);
  }

  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus("IsNaN", elem_type);
  }

  musaError_t status = LaunchMusaIsNaNKernel(
      input.GetTensorRawData(), y.GetTensorMutableData<uint8_t>(),
      NumElements(shape), musa_elem_type, GetComputeStream(ctx));
  if (status == musaErrorNotSupported) {
    return UnsupportedDeviceElementwiseStatus("IsNaN", elem_type);
  }
  return LaunchStatus(status);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    IsNaN, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T1", FloatBoolTensorTypes())
         .AddTypeConstraint("T2",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))),
    IsNaN)
