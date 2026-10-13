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

#include "graph/capability_predicates.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "graph/graph_utils.h"
#include "tensor/resize_support.h"

namespace musa_ep {
namespace {

/// Reads a constant float initializer. Returns nullopt when the value is not a
/// constant float tensor, i.e. when only the runtime kernel can judge it.
std::optional<std::vector<float>> ReadConstantFloatTensor(
    Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    return std::nullopt;
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    return std::nullopt;
  }

  const auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return std::nullopt;
  }
  const size_t element_count = static_cast<size_t>(info.GetElementCount());
  if (element_count == 0) {
    return std::vector<float>();
  }
  const float* data = value.GetTensorData<float>();
  return std::vector<float>(data, data + element_count);
}

/// Element count of a constant initializer, or nullopt when the tensor only
/// exists at run time.
std::optional<int64_t> ConstantTensorElementCount(
    Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    return std::nullopt;
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    return std::nullopt;
  }

  return value.GetTensorTypeAndShapeInfo().GetElementCount();
}

/// True when the node leaves the optional input at \p index unused, which is
/// the case when it is omitted or when the graph fills it with a zero-element
/// constant: exporters that drive Resize through `sizes` commonly pass an empty
/// `roi` and `scales` initializer next to it, and the kernel reads those as
/// absent as well. A run-time tensor is assumed to be used, because only the
/// kernel can see its element count.
bool InputUnused(const std::vector<Ort::ConstValueInfo>& inputs, size_t index) {
  if (inputs.size() <= index || inputs[index] == nullptr) {
    return true;
  }
  return ResizeInputUnused(ConstantTensorElementCount(inputs[index]));
}

/// True when the MUSA Resize kernel can run this node. Attribute values, which
/// optional input is supplied and the type, rank and shape of X decide the
/// answer for every node; the values of the roi/scales/sizes tensors are
/// checked as well when the graph carries them as constant initializers, and
/// are otherwise left to the kernel.
bool ResizeConfigurationSupported(Ort::ConstNode node) {
  ResizeRouteConfig config;
  config.mode = GetStringAttribute(node, "mode").value_or(kResizeDefaultMode);
  config.coordinate_transformation_mode =
      GetStringAttribute(node, "coordinate_transformation_mode")
          .value_or(kResizeDefaultCoordinateTransformationMode);
  config.nearest_mode = GetStringAttribute(node, "nearest_mode")
                            .value_or(kResizeDefaultNearestMode);

  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  if (inputs.empty() || inputs[0] == nullptr) {
    return false;
  }
  config.has_scales_input = !InputUnused(inputs, 2);
  config.has_sizes_input = !InputUnused(inputs, 3);

  const Ort::ConstTypeInfo input_type = inputs[0].TypeInfo();
  if (input_type.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  const auto input_info = input_type.GetTensorTypeAndShapeInfo();
  config.input_rank = static_cast<int64_t>(input_info.GetShape().size());
  config.input_element_type = input_info.GetElementType();
  if (ResizeRouteRejection(config) != nullptr) {
    return false;
  }

  ResizeRouteValues values;
  if (inputs.size() > 1) {
    values.roi_element_count = ConstantTensorElementCount(inputs[1]);
  }
  if (config.has_scales_input) {
    values.scales = ReadConstantFloatTensor(inputs[2]);
  }
  if (config.has_sizes_input) {
    values.sizes = ReadIntInitializerNoLimit(inputs[3]);
  }
  values.input_shape = GetStaticShape(inputs[0]);
  return ResizeRouteValueRejection(config, values) == nullptr;
}

}  // namespace

bool IsNodeConfigurationSupported(Ort::ConstNode node) {
  if (IsOnnxOp(node, "Resize")) {
    return ResizeConfigurationSupported(node);
  }
  return true;
}

}  // namespace musa_ep
