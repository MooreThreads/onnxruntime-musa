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
#include "tensor/resize_impl.h"

namespace {

class Resize : public OpKernelBase<Resize> {
 public:
  Resize(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    mode_ = AttrOrDefault<std::string>(kernel_info, "mode", "nearest");
    coordinate_transform_mode_ = AttrOrDefault<std::string>(
        kernel_info, "coordinate_transformation_mode", "half_pixel");
    nearest_mode_ = AttrOrDefault<std::string>(kernel_info, "nearest_mode",
                                               "round_prefer_floor");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  std::string mode_;
  std::string coordinate_transform_mode_;
  std::string nearest_mode_;
};

OrtStatus* Resize::Compute(Ort::KernelContext& ctx) const {
  if (mode_ != "nearest" || coordinate_transform_mode_ != "asymmetric" ||
      nearest_mode_ != "floor") {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Resize only supports nearest/asymmetric/floor in this MUSA route");
  }
  if (ctx.GetInputCount() <= 2 || ctx.GetInput(2) == nullptr ||
      (ctx.GetInputCount() > 3 && ctx.GetInput(3) != nullptr)) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Resize only supports the scales input, not the sizes input");
  }
  if (ctx.GetInputCount() > 1 && ctx.GetInput(1) != nullptr &&
      ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetElementCount() != 0) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Resize only supports an empty ROI for asymmetric coordinates");
  }

  Ort::ConstValue input = ctx.GetInput(0);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Resize only supports float32");
  }
  const std::vector<int64_t> input_shape = input_info.GetShape();
  if (input_shape.size() != 4 || input_shape[2] <= 0 || input_shape[3] <= 0) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Resize only supports non-empty rank-4 NCHW tensors");
  }
  const std::vector<float> scales =
      ReadTyped<float>(ctx.GetInput(2), GetComputeStream(ctx));
  if (scales.size() != 4 || scales[0] != 1.0f || scales[1] != 1.0f ||
      !std::isfinite(scales[2]) || !std::isfinite(scales[3]) ||
      scales[2] <= 0.0f || scales[3] <= 0.0f) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Resize requires positive NCHW scales with batch/channel scale 1");
  }

  std::vector<int64_t> output_shape = input_shape;
  output_shape[2] =
      static_cast<int64_t>(std::floor(input_shape[2] * scales[2]));
  output_shape[3] =
      static_cast<int64_t>(std::floor(input_shape[3] * scales[3]));
  if (output_shape[2] <= 0 || output_shape[3] <= 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Resize output must be non-empty");
  }
  if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Resize requires MUSA input");
  }
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Resize requires MUSA output");
  }

  return LaunchStatus(LaunchMusaResizeNearest2DFloat(
      input.GetTensorData<float>(), output.GetTensorMutableData<float>(),
      input_shape[0], input_shape[1], input_shape[2], input_shape[3],
      output_shape[2], output_shape[3], scales[2], scales[3],
      GetComputeStream(ctx)));
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Resize, kOnnxDomain, 13, 17,
                                  (Ort::KernelDefBuilder()
                                       .AddTypeConstraint("T1",
                                                          FloatTensorTypes())
                                       .SetInputMemType(1, OrtMemTypeCPUInput)
                                       .SetInputMemType(2, OrtMemTypeCPUInput)
                                       .SetInputMemType(3, OrtMemTypeCPUInput)),
                                  Resize)
