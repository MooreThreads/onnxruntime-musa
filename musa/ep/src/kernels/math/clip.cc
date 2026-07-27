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

#include "math/clip_impl.h"
#include "shared_inc/op_kernel_common.h"

namespace {

OrtStatus* ValidateScalarInput(Ort::ConstValue value,
                               ONNXTensorElementDataType elem_type,
                               const char* input_name) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != elem_type || info.GetElementCount() != 1) {
    const std::string message = std::string("Clip ") + input_name +
                                " must be a scalar matching X dtype";
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, message.c_str());
  }
  const size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Clip unsupported scalar dtype");
  }
  return nullptr;
}

class Clip : public OpKernelBase<Clip> {
 public:
  Clip(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Clip::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto elem_type = input_info.GetElementType();
  auto shape = input_info.GetShape();
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Clip unsupported dtype");
  }
  musaStream_t stream = GetComputeStream(ctx);

  MusaClipParams params{};
  params.count = NumElements(shape);
  DeviceInputBuffer min_buffer;
  DeviceInputBuffer max_buffer;
  if (ctx.GetInputCount() > 1 && ctx.GetInput(1) != nullptr) {
    RETURN_IF_ERROR(ValidateScalarInput(ctx.GetInput(1), elem_type, "min"));
    RETURN_IF_ERROR(min_buffer.Bind(ctx.GetInput(1), stream));
    params.has_min = 1;
    params.min_data = min_buffer.data();
  }
  if (ctx.GetInputCount() > 2 && ctx.GetInput(2) != nullptr) {
    RETURN_IF_ERROR(ValidateScalarInput(ctx.GetInput(2), elem_type, "max"));
    RETURN_IF_ERROR(max_buffer.Bind(ctx.GetInput(2), stream));
    params.has_max = 1;
    params.max_data = max_buffer.data();
  }

  if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Clip requires MUSA input");
  }
  Ort::UnownedValue output = ctx.GetOutput(0, shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Clip requires MUSA output");
  }

  musaError_t status = LaunchMusaClipKernel(input.GetTensorRawData(),
                                            output.GetTensorMutableRawData(),
                                            params, musa_elem_type, stream);
  if (status == musaErrorNotSupported) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Clip unsupported dtype");
  }
  return LaunchStatus(status);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Clip, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", ClipTensorTypes())), Clip)
