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

#include <cmath>
#include <string>
#include <vector>

#include "shared_inc/op_kernel_common.h"
#include "tensor/resize_impl.h"
#include "tensor/resize_support.h"

namespace {

OrtStatus* NotImplemented(const char* message) {
  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, message);
}

bool ParseCoordinateMode(const std::string& name, int32_t& mode) {
  if (name == "half_pixel") {
    mode = kMusaResizeHalfPixel;
    return true;
  }
  if (name == "pytorch_half_pixel") {
    mode = kMusaResizePytorchHalfPixel;
    return true;
  }
  if (name == "align_corners") {
    mode = kMusaResizeAlignCorners;
    return true;
  }
  if (name == "asymmetric") {
    mode = kMusaResizeAsymmetric;
    return true;
  }
  return false;
}

// An optional Resize input is unused when it is absent or carries no elements
// (an empty roi/scales tensor is how the exporters mark "not provided"); the
// graph-level capability predicate applies the same rule to the constant
// initializers through musa_ep::ResizeInputUnused().
bool IsAbsentInput(Ort::KernelContext& ctx, size_t index) {
  if (ctx.GetInputCount() <= index) {
    return true;
  }
  Ort::ConstValue value = ctx.GetInput(index);
  if (value == nullptr) {
    return true;
  }
  return musa_ep::ResizeInputUnused(
      value.GetTensorTypeAndShapeInfo().GetElementCount());
}

class Resize : public OpKernelBase<Resize> {
 public:
  Resize(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    config_.mode = AttrOrDefault<std::string>(kernel_info, "mode",
                                              musa_ep::kResizeDefaultMode);
    config_.coordinate_transformation_mode = AttrOrDefault<std::string>(
        kernel_info, "coordinate_transformation_mode",
        musa_ep::kResizeDefaultCoordinateTransformationMode);
    config_.nearest_mode = AttrOrDefault<std::string>(
        kernel_info, "nearest_mode", musa_ep::kResizeDefaultNearestMode);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  musa_ep::ResizeRouteConfig config_;
};

OrtStatus* Resize::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  const std::vector<int64_t> input_shape = input_info.GetShape();

  // Route checks shared with the graph-level capability predicate in
  // graph/capability_predicates.cc: a configuration this route cannot run must
  // not be claimed by the EP, and must be rejected here if it was claimed.
  musa_ep::ResizeRouteConfig config = config_;
  config.has_scales_input = !IsAbsentInput(ctx, 2);
  config.has_sizes_input = !IsAbsentInput(ctx, 3);
  config.input_rank = static_cast<int64_t>(input_shape.size());
  config.input_element_type = input_info.GetElementType();
  if (const char* reason = musa_ep::ResizeRouteRejection(config);
      reason != nullptr) {
    return NotImplemented(reason);
  }

  const size_t rank = input_shape.size();
  for (int64_t dim : input_shape) {
    if (dim <= 0) {
      return NotImplemented("Resize requires a non-empty input tensor");
    }
  }

  musaStream_t stream = GetComputeStream(ctx);
  musa_ep::ResizeRouteValues values;
  if (ctx.GetInputCount() > 1 && ctx.GetInput(1) != nullptr) {
    values.roi_element_count =
        ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetElementCount();
  }
  if (config.has_scales_input) {
    values.scales = ReadTyped<float>(ctx.GetInput(2), stream);
  } else {
    values.sizes = ReadIntTensor(ctx, 3);
  }
  values.input_shape = input_shape;
  if (const char* reason = musa_ep::ResizeRouteValueRejection(config, values);
      reason != nullptr) {
    return NotImplemented(reason);
  }

  // Output shape and the effective per-axis scale factor. This mirrors the ORT
  // CPU kernel: with `scales` the factor is the value itself, with `sizes` it
  // is output/input (see ParseScalesDataAndAdjustOutputSize).
  std::vector<int64_t> output_shape(rank);
  std::vector<float> scales(rank);
  if (config.has_scales_input) {
    scales = *values.scales;
    for (size_t axis = 0; axis < rank; ++axis) {
      output_shape[axis] = static_cast<int64_t>(
          scales[axis] * static_cast<float>(input_shape[axis]));
      if (output_shape[axis] <= 0) {
        return NotImplemented("Resize output must be non-empty");
      }
    }
  } else {
    const std::vector<int64_t>& sizes = *values.sizes;
    for (size_t axis = 0; axis < rank; ++axis) {
      output_shape[axis] = sizes[axis];
      scales[axis] = static_cast<float>(sizes[axis]) /
                     static_cast<float>(input_shape[axis]);
    }
  }

  int32_t coordinate_mode = 0;
  const bool coordinate_mode_known = ParseCoordinateMode(
      config.coordinate_transformation_mode, coordinate_mode);
  if (config.mode == "linear" && !coordinate_mode_known) {
    // Unreachable while ResizeRouteRejection() rejects every coordinate
    // transformation mode that ParseCoordinateMode() does not know.
    return NotImplemented(musa_ep::kResizeLinearCoordinateReason);
  }

  if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
    return NotImplemented("Resize requires MUSA input");
  }
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return NotImplemented("Resize requires MUSA output");
  }

  if (config.mode == "nearest") {
    return LaunchStatus(LaunchMusaResizeNearest2DFloat(
        input.GetTensorData<float>(), output.GetTensorMutableData<float>(),
        input_shape[0], input_shape[1], input_shape[2], input_shape[3],
        output_shape[2], output_shape[3], scales[2], scales[3], stream));
  }
  if (rank == 4) {
    return LaunchStatus(LaunchMusaResizeLinear2DFloat(
        input.GetTensorData<float>(), output.GetTensorMutableData<float>(),
        input_shape[0], input_shape[1], input_shape[2], input_shape[3],
        output_shape[2], output_shape[3], scales[2], scales[3], coordinate_mode,
        stream));
  }
  return LaunchStatus(LaunchMusaResizeLinear3DFloat(
      input.GetTensorData<float>(), output.GetTensorMutableData<float>(),
      input_shape[0], input_shape[1], input_shape[2], input_shape[3],
      input_shape[4], output_shape[2], output_shape[3], output_shape[4],
      scales[2], scales[3], scales[4], coordinate_mode, stream));
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
