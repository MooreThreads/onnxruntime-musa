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

#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "onnxruntime_c_api.h"

namespace musa_ep {

// ONNX Resize (opset 13+) attribute defaults.
inline constexpr char kResizeDefaultMode[] = "nearest";
inline constexpr char kResizeDefaultCoordinateTransformationMode[] =
    "half_pixel";
inline constexpr char kResizeDefaultNearestMode[] = "round_prefer_floor";

/// A zero-element optional input means "not provided": that is how the
/// exporters mark an unused roi/scales, so an omitted input and an empty one
/// are the same thing for this route. An unknown element count (a run-time
/// tensor seen from the graph) must count as provided, leaving the decision to
/// the kernel.
inline bool ResizeInputUnused(std::optional<int64_t> element_count) {
  return element_count.has_value() && *element_count == 0;
}

/// The coordinate transformations the linear route implements.
inline bool IsSupportedResizeLinearCoordinateMode(const std::string& mode) {
  return mode == "half_pixel" || mode == "pytorch_half_pixel" ||
         mode == "align_corners" || mode == "asymmetric";
}

/// Reason string for the coordinate transformation modes the linear route does
/// not implement; shared so the graph-level gate and the kernel report the same
/// thing.
inline constexpr char kResizeLinearCoordinateReason[] =
    "Resize linear only supports half_pixel, pytorch_half_pixel, "
    "align_corners or asymmetric coordinates in this MUSA route";

/// Everything about a Resize node that decides whether this MUSA route can run
/// it, apart from the contents of its tensors.
///
/// The kernel registry only matches op/domain/opset/dtype, so the graph-level
/// capability predicate (graph/capability_predicates.cc) inspects the node
/// itself to keep the Resize configurations this route cannot run on the CPU
/// EP. Both it and the kernel's runtime guard build this struct and ask
/// ResizeRouteRejection() below, so the nodes this route claims and the nodes
/// it runs are decided by one definition instead of two copies that drift
/// apart.
struct ResizeRouteConfig {
  std::string mode = kResizeDefaultMode;
  std::string coordinate_transformation_mode =
      kResizeDefaultCoordinateTransformationMode;
  std::string nearest_mode = kResizeDefaultNearestMode;
  bool has_scales_input = false;
  bool has_sizes_input = false;
  /// Rank of the X input; -1 when it is unknown or not a tensor.
  int64_t input_rank = -1;
  ONNXTensorElementDataType input_element_type =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
};

/// Tensor contents this route constrains. Callers only fill in what they can
/// observe - the capability predicate knows the constant initializers, the
/// kernel the actual runtime tensors. An unset field is not checked here and
/// stays the runtime guard's job.
struct ResizeRouteValues {
  /// Element count of the (optional) roi input. Only an empty roi is supported.
  std::optional<int64_t> roi_element_count;
  /// Contents of the scales input.
  std::optional<std::vector<float>> scales;
  /// Contents of the sizes input.
  std::optional<std::vector<int64_t>> sizes;
  /// Shape of X, compared against sizes to tell whether the batch and channel
  /// axes are resized.
  std::optional<std::vector<int64_t>> input_shape;
};

/// Returns nullptr when this MUSA route runs \p config, otherwise a static
/// reason string suitable for an ORT_NOT_IMPLEMENTED status.
inline const char* ResizeRouteRejection(const ResizeRouteConfig& config) {
  if (config.mode != "nearest" && config.mode != "linear") {
    return "Resize only supports mode nearest or linear in this MUSA route";
  }
  if (config.input_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return "Resize only supports float32";
  }
  if (config.has_scales_input == config.has_sizes_input) {
    return "Resize requires exactly one of the scales or sizes input";
  }
  if (config.input_rank != 4 && config.input_rank != 5) {
    return "Resize only supports rank-4 NCHW and rank-5 NCDHW tensors in this "
           "MUSA route";
  }
  if (config.mode == "linear") {
    if (!IsSupportedResizeLinearCoordinateMode(
            config.coordinate_transformation_mode)) {
      return kResizeLinearCoordinateReason;
    }
    return nullptr;
  }
  if (config.coordinate_transformation_mode != "asymmetric" ||
      config.nearest_mode != "floor") {
    return "Resize only supports nearest/asymmetric/floor in this MUSA route";
  }
  if (config.input_rank != 4) {
    return "Resize nearest only supports rank-4 NCHW tensors in this MUSA "
           "route";
  }
  return nullptr;
}

/// Value-dependent counterpart of ResizeRouteRejection().
inline const char* ResizeRouteValueRejection(const ResizeRouteConfig& config,
                                             const ResizeRouteValues& values) {
  if (values.roi_element_count.has_value() && *values.roi_element_count != 0) {
    return "Resize only supports an empty ROI (tf_crop_and_resize is not "
           "implemented)";
  }
  if (config.has_scales_input && values.scales.has_value()) {
    const std::vector<float>& scales = *values.scales;
    if (config.input_rank > 0 &&
        static_cast<int64_t>(scales.size()) != config.input_rank) {
      return "Resize scales rank must equal the input rank";
    }
    for (float scale : scales) {
      if (!std::isfinite(scale) || scale <= 0.0f) {
        return "Resize requires positive finite scales";
      }
    }
    if (scales.size() >= 2 && (scales[0] != 1.0f || scales[1] != 1.0f)) {
      return "Resize requires batch and channel scales to be 1 in this MUSA "
             "route";
    }
  }
  if (config.has_sizes_input && values.sizes.has_value()) {
    const std::vector<int64_t>& sizes = *values.sizes;
    if (config.input_rank > 0 &&
        static_cast<int64_t>(sizes.size()) != config.input_rank) {
      return "Resize sizes rank must equal the input rank";
    }
    for (int64_t size : sizes) {
      if (size <= 0) {
        return "Resize requires positive sizes";
      }
    }
    // `sizes` resizes the batch and channel axes when it disagrees with the
    // static shape of X; the scales rule above states the same constraint.
    if (values.input_shape.has_value() &&
        values.input_shape->size() == sizes.size() && sizes.size() >= 2) {
      const std::vector<int64_t>& shape = *values.input_shape;
      const bool resizes_batch = shape[0] > 0 && sizes[0] != shape[0];
      const bool resizes_channel = shape[1] > 0 && sizes[1] != shape[1];
      if (resizes_batch || resizes_channel) {
        return "Resize requires batch and channel scales to be 1 in this MUSA "
               "route";
      }
    }
  }
  return nullptr;
}

}  // namespace musa_ep
