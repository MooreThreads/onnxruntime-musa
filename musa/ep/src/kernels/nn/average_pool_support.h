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

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "onnxruntime_c_api.h"

namespace musa_ep {

// Mirrors kMusaMaxBroadcastRank in ../shared_inc/device_kernel_types.h; kept
// local so this header (also included by the graph-level capability predicate)
// depends on the public C API only.
inline constexpr int64_t kAveragePoolMaxRank = 8;

inline constexpr char kAveragePoolDefaultAutoPad[] = "NOTSET";

/// Everything about an AveragePool node that decides whether this MUSA route
/// can run it.
///
/// The kernel registry only matches op/domain/opset/dtype, so the graph-level
/// capability predicate (graph/capability_predicates.cc) inspects the node
/// itself to keep the AveragePool configurations this route cannot run on the
/// CPU EP. Both it and the kernel's runtime guard build this struct and ask
/// AveragePoolRouteRejection() below, so the nodes this route claims and the
/// nodes it runs are decided by one definition instead of two copies that
/// drift apart.
struct AveragePoolRouteConfig {
  std::string auto_pad = kAveragePoolDefaultAutoPad;
  int64_t ceil_mode = 0;
  int64_t count_include_pad = 0;
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
  /// Rank of the X input; -1 when it is unknown or not a tensor.
  int64_t input_rank = -1;
  ONNXTensorElementDataType input_element_type =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
};

/// Returns nullptr when this MUSA route runs \p config, otherwise a static
/// reason string suitable for an ORT_NOT_IMPLEMENTED status.
inline const char* AveragePoolRouteRejection(
    const AveragePoolRouteConfig& config) {
  if (config.auto_pad != "NOTSET" && config.auto_pad != "VALID" &&
      config.auto_pad != "SAME_UPPER" && config.auto_pad != "SAME_LOWER") {
    return "AveragePool only supports auto_pad NOTSET, VALID, SAME_UPPER or "
           "SAME_LOWER in this MUSA route";
  }
  // ceil_mode=1 changes the output size and clips the last window, which the
  // pooling routes here do not model; those nodes stay on the CPU EP.
  if (config.ceil_mode != 0) {
    return "AveragePool only supports ceil_mode=0 in this MUSA route";
  }
  if (config.count_include_pad != 0 && config.count_include_pad != 1) {
    return "AveragePool requires count_include_pad to be 0 or 1";
  }
  if (config.input_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      config.input_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
      config.input_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    return "AveragePool only supports float16, float32 or float64";
  }
  if (config.input_rank < 3 || config.input_rank > kAveragePoolMaxRank) {
    return "AveragePool only supports rank 3 to 8 tensors in this MUSA route";
  }
  const auto spatial_rank = static_cast<size_t>(config.input_rank) - 2;
  if (config.kernel_shape.size() != spatial_rank) {
    return "AveragePool requires kernel_shape with one entry per spatial "
           "dimension";
  }
  if (!config.strides.empty() && config.strides.size() != spatial_rank) {
    return "AveragePool requires strides with one entry per spatial dimension";
  }
  // The auto_pad modes derive the pads from the input shape, so a pads
  // attribute next to them is ignored (ORT does the same); only auto_pad
  // NOTSET reads the attribute.
  if (config.auto_pad == kAveragePoolDefaultAutoPad && !config.pads.empty() &&
      config.pads.size() != spatial_rank * 2) {
    return "AveragePool requires pads with two entries per spatial dimension";
  }
  for (int64_t value : config.kernel_shape) {
    if (value <= 0) {
      return "AveragePool requires a positive kernel_shape";
    }
  }
  for (int64_t value : config.strides) {
    if (value <= 0) {
      return "AveragePool requires positive strides";
    }
  }
  if (config.auto_pad == kAveragePoolDefaultAutoPad) {
    for (int64_t value : config.pads) {
      if (value < 0) {
        return "AveragePool requires non-negative pads";
      }
    }
  }
  return nullptr;
}

/// Pads this route runs with: the attribute for auto_pad NOTSET, zeros for
/// VALID, and the pads ONNX derives from the input size for the SAME_* modes
/// (the output size is ceil(input / stride), so the total pad is
/// (output - 1) * stride + kernel - input). Returns false when a spatial
/// dimension is unknown, which only the auto_pad modes need.
inline bool ResolveAveragePoolPads(std::string_view auto_pad,
                                   const std::vector<int64_t>& pads_attr,
                                   const std::vector<int64_t>& input_shape,
                                   const std::vector<int64_t>& kernel_shape,
                                   const std::vector<int64_t>& strides,
                                   std::vector<int64_t>& pads) {
  const size_t spatial_rank = kernel_shape.size();
  pads.assign(spatial_rank * 2, 0);
  if (auto_pad == kAveragePoolDefaultAutoPad) {
    if (pads_attr.size() == spatial_rank * 2) {
      pads = pads_attr;
    }
    return true;
  }
  if (auto_pad == "VALID") {
    return true;
  }
  if (input_shape.size() != spatial_rank + 2) {
    return false;
  }
  for (size_t i = 0; i < spatial_rank; ++i) {
    const int64_t input_dim = input_shape[i + 2];
    if (input_dim <= 0) {
      return false;
    }
    const int64_t output_dim = (input_dim + strides[i] - 1) / strides[i];
    const int64_t total_pad = std::max<int64_t>(
        0, (output_dim - 1) * strides[i] + kernel_shape[i] - input_dim);
    const int64_t pad_begin =
        auto_pad == "SAME_LOWER" ? (total_pad + 1) / 2 : total_pad / 2;
    pads[i] = pad_begin;
    pads[i + spatial_rank] = total_pad - pad_begin;
  }
  return true;
}

}  // namespace musa_ep
