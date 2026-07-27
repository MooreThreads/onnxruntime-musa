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

#include "graph/graph_utils.h"

#include "plugin_ep_utils.h"

namespace musa_ep {

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(const Ort::ConstNode& node, const char* op_type) {
  return node && node.GetOperatorType() == op_type &&
         IsOnnxDomain(node.GetDomain());
}

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
}

bool IsFloatTensorValueInfo(Ort::ConstValueInfo value_info) {
  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }

  auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  return type_shape.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

bool IsConstantInitializerValueInfo(Ort::ConstValueInfo value_info) {
  return value_info != nullptr && value_info.IsConstantInitializer();
}

bool IsIntTensorValueInfo(Ort::ConstValueInfo value_info) {
  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }

  auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType elem_type = type_shape.GetElementType();
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

bool IsSmallInitializer(Ort::ConstValueInfo input) {
  if (!input.IsConstantInitializer()) {
    return false;
  }
  Ort::ConstTypeInfo type_info = input.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  return type_info.GetTensorTypeAndShapeInfo().GetElementCount() <=
         kSmallInitializerThreshold;
}

std::optional<std::vector<int64_t>> GetStaticShape(
    Ort::ConstValueInfo value_info) {
  auto shape = GetTensorShape(value_info);
  if (!shape.has_value()) {
    return std::nullopt;
  }

  for (int64_t dim : *shape) {
    if (dim <= 0) {
      return std::nullopt;
    }
  }

  return shape;
}

std::optional<int64_t> GetIntAttribute(Ort::ConstNode node,
                                       const std::string& name) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK()) {
    return std::nullopt;
  }

  int64_t value = 0;
  status = attr.GetValue(value);
  if (!status.IsOK()) {
    return std::nullopt;
  }

  return value;
}

std::optional<std::vector<int64_t>> GetIntsAttribute(Ort::ConstNode node,
                                                     const std::string& name) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK()) {
    return std::nullopt;
  }

  std::vector<int64_t> values;
  status = attr.GetValueArray<int64_t>(values);
  if (!status.IsOK()) {
    return std::nullopt;
  }
  return values;
}

std::optional<std::string> GetStringAttribute(Ort::ConstNode node,
                                              const std::string& name) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK()) {
    return std::nullopt;
  }

  std::string value;
  status = attr.GetValue(value);
  if (!status.IsOK()) {
    return std::nullopt;
  }
  return value;
}

bool NormalizeAxis(int64_t axis, size_t rank, int64_t& normalized_axis) {
  const int64_t signed_rank = static_cast<int64_t>(rank);
  if (rank == 0 || axis < -signed_rank || axis >= signed_rank) {
    return false;
  }

  normalized_axis = axis < 0 ? axis + signed_rank : axis;
  return true;
}

bool KnownDimsEqual(int64_t lhs, int64_t rhs) {
  return lhs <= 0 || rhs <= 0 || lhs == rhs;
}

bool ShapesEqualOnKnownDims(const std::vector<int64_t>& lhs,
                            const std::vector<int64_t>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!KnownDimsEqual(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

std::optional<std::vector<int64_t>> ReadSmallIntInitializer(
    Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    return std::nullopt;
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    return std::nullopt;
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  const size_t count = static_cast<size_t>(info.GetElementCount());
  if (count > kSmallInitializerThreshold) {
    return std::nullopt;
  }

  std::vector<int64_t> result;
  result.reserve(count);
  ONNXTensorElementDataType elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t* data = value.GetTensorData<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      result.push_back(static_cast<int64_t>(data[i]));
    }
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* data = value.GetTensorData<int64_t>();
    result.assign(data, data + count);
  } else {
    return std::nullopt;
  }
  return result;
}

std::optional<std::vector<int64_t>> ReadIntInitializerNoLimit(
    Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    return std::nullopt;
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    return std::nullopt;
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  const size_t count = static_cast<size_t>(info.GetElementCount());
  std::vector<int64_t> result;
  result.reserve(count);
  ONNXTensorElementDataType elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t* data = value.GetTensorData<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      result.push_back(static_cast<int64_t>(data[i]));
    }
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* data = value.GetTensorData<int64_t>();
    result.assign(data, data + count);
  } else {
    return std::nullopt;
  }
  return result;
}

std::optional<float> ReadScalarFloatInitializer(
    Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    return std::nullopt;
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    return std::nullopt;
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      info.GetElementCount() != 1) {
    return std::nullopt;
  }

  return value.GetTensorData<float>()[0];
}

std::optional<std::vector<int64_t>> ReadUnsqueezeAxes(
    Ort::ConstNode unsqueeze_node) {
  std::vector<Ort::ConstValueInfo> inputs = unsqueeze_node.GetInputs();
  if (inputs.size() >= 2) {
    return ReadIntInitializerNoLimit(inputs[1]);
  }
  return GetIntsAttribute(unsqueeze_node, "axes");
}

bool IsZeroFloatConstantOfShape(Ort::ConstNode node) {
  if (!IsOnnxOp(node, "ConstantOfShape")) {
    return false;
  }

  Ort::ConstOpAttr attr;
  Ort::Status attr_status = node.GetAttributeByName("value", attr);
  if (!attr_status.IsOK()) {
    return true;
  }

  Ort::Value value{nullptr};
  Ort::Status status = attr.GetTensorAttributeAsOrtValue(value);
  if (!status.IsOK() || !value) {
    return false;
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      info.GetElementCount() != 1) {
    return false;
  }
  return value.GetTensorData<float>()[0] == 0.0f;
}

std::optional<std::vector<int64_t>> ConstantOfShapeOutputShape(
    Ort::ConstNode node, Ort::ConstValueInfo output_info) {
  auto output_shape = GetTensorShape(output_info);
  if (output_shape.has_value() && output_shape->size() == 2 &&
      (*output_shape)[1] > 0) {
    return output_shape;
  }

  std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  if (inputs.size() != 1) {
    return std::nullopt;
  }

  auto shape_from_initializer = ReadIntInitializerNoLimit(inputs[0]);
  if (shape_from_initializer.has_value() &&
      shape_from_initializer->size() == 2 && (*shape_from_initializer)[1] > 0) {
    return shape_from_initializer;
  }

  Ort::ValueInfoConsumerProducerInfo producer = inputs[0].GetProducerNode();
  if (!producer.node || !IsOnnxOp(producer.node, "Shape")) {
    return std::nullopt;
  }

  std::vector<Ort::ConstValueInfo> shape_inputs = producer.node.GetInputs();
  if (shape_inputs.size() != 1) {
    return std::nullopt;
  }
  auto shape = GetTensorShape(shape_inputs[0]);
  if (!shape.has_value() || shape->size() != 2 || (*shape)[1] <= 0) {
    return std::nullopt;
  }
  return shape;
}

}  // namespace musa_ep
