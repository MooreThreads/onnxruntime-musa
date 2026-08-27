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
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "fusion/mhta_scaled_dot_product_attention_utils.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

bool HasSingleConsumer(Ort::ConstValueInfo value,
                       const std::unordered_set<std::string>& graph_outputs) {
  return graph_outputs.count(Name(value)) == 0 &&
         value.GetConsumers().size() == 1;
}

bool IsZeroFloatInitializer(Ort::ConstValueInfo value_info) {
  if (!value_info.IsConstantInitializer() ||
      value_info.TypeInfo().GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK()) {
    return false;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  const ONNXTensorElementDataType elem_type = info.GetElementType();
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    return false;
  }
  const size_t count = info.GetElementCount();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float* data = value.GetTensorData<float>();
    for (size_t i = 0; i < count; ++i) {
      if (std::fabs(data[i]) > 0.0f) return false;
    }
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    const double* data = value.GetTensorData<double>();
    for (size_t i = 0; i < count; ++i) {
      if (data[i] != 0.0) return false;
    }
  } else {
    const uint16_t* data = value.GetTensorData<uint16_t>();
    for (size_t i = 0; i < count; ++i) {
      if (data[i] != 0) return false;
    }
  }
  return true;
}

bool IsMhtaSdpaDataType(Ort::ConstValueInfo value_info) {
  if (value_info.TypeInfo().GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  const ONNXTensorElementDataType elem_type =
      value_info.TypeInfo().GetTensorTypeAndShapeInfo().GetElementType();
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}

bool HasSameMhtaSdpaDataType(Ort::ConstValueInfo lhs, Ort::ConstValueInfo rhs) {
  return IsMhtaSdpaDataType(lhs) && IsMhtaSdpaDataType(rhs) &&
         lhs.TypeInfo().GetTensorTypeAndShapeInfo().GetElementType() ==
             rhs.TypeInfo().GetTensorTypeAndShapeInfo().GetElementType();
}

bool IsBoolTensorValueInfo(Ort::ConstValueInfo value_info) {
  return value_info.TypeInfo().GetONNXType() == ONNX_TYPE_TENSOR &&
         value_info.TypeInfo().GetTensorTypeAndShapeInfo().GetElementType() ==
             ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
}

bool IsInt32TensorValueInfo(Ort::ConstValueInfo value_info) {
  return value_info.TypeInfo().GetONNXType() == ONNX_TYPE_TENSOR &&
         value_info.TypeInfo().GetTensorTypeAndShapeInfo().GetElementType() ==
             ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}

bool IsInt64TensorValueInfo(Ort::ConstValueInfo value_info) {
  return value_info.TypeInfo().GetONNXType() == ONNX_TYPE_TENSOR &&
         value_info.TypeInfo().GetTensorTypeAndShapeInfo().GetElementType() ==
             ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

bool IsFloat32TensorValueInfo(Ort::ConstValueInfo value_info) {
  return value_info.TypeInfo().GetONNXType() == ONNX_TYPE_TENSOR &&
         value_info.TypeInfo().GetTensorTypeAndShapeInfo().GetElementType() ==
             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

bool IsBfloat16TensorValueInfo(Ort::ConstValueInfo value_info) {
  return value_info.TypeInfo().GetONNXType() == ONNX_TYPE_TENSOR &&
         value_info.TypeInfo().GetTensorTypeAndShapeInfo().GetElementType() ==
             ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
}

bool IsExactTranspose(Ort::ConstNode node,
                      const std::vector<int64_t>& expected_perm) {
  const auto perm = GetIntsAttribute(node, "perm");
  return IsOnnxOp(node, "Transpose") && perm.has_value() &&
         *perm == expected_perm;
}

bool IsBshdBoundaryShape(const std::vector<int64_t>& shape) {
  return shape.size() == 4;
}

bool BshdBoundaryShapesAreSupported(Ort::ConstValueInfo q,
                                    Ort::ConstValueInfo k,
                                    Ort::ConstValueInfo v,
                                    Ort::ConstValueInfo output) {
  if (!HasSameMhtaSdpaDataType(q, k) || !HasSameMhtaSdpaDataType(q, v) ||
      !HasSameMhtaSdpaDataType(q, output)) {
    return false;
  }
  const auto elem_type =
      q.TypeInfo().GetTensorTypeAndShapeInfo().GetElementType();
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    return false;
  }
  const auto q_shape = GetTensorShape(q);
  const auto k_shape = GetTensorShape(k);
  const auto v_shape = GetTensorShape(v);
  const auto output_shape = GetTensorShape(output);
  if (!q_shape.has_value() || !k_shape.has_value() || !v_shape.has_value() ||
      !output_shape.has_value() || !IsBshdBoundaryShape(*q_shape) ||
      !IsBshdBoundaryShape(*k_shape) || !IsBshdBoundaryShape(*v_shape) ||
      !IsBshdBoundaryShape(*output_shape)) {
    return false;
  }

  // The four absorbed nodes expose ordinary contiguous [B,S,H,D] buffers at
  // the fusion boundary. The SDPA core still consumes logical [B,H,S,D].
  return KnownDimsEqual((*q_shape)[0], (*k_shape)[0]) &&
         KnownDimsEqual((*q_shape)[0], (*v_shape)[0]) &&
         KnownDimsEqual((*q_shape)[1], (*k_shape)[1]) &&
         KnownDimsEqual((*q_shape)[1], (*v_shape)[1]) &&
         KnownDimsEqual((*q_shape)[2], (*k_shape)[2]) &&
         KnownDimsEqual((*q_shape)[2], (*v_shape)[2]) &&
         KnownDimsEqual((*q_shape)[3], (*k_shape)[3]) &&
         KnownDimsEqual((*q_shape)[3], (*v_shape)[3]) &&
         KnownDimsEqual((*output_shape)[0], (*q_shape)[0]) &&
         KnownDimsEqual((*output_shape)[1], (*q_shape)[1]) &&
         KnownDimsEqual((*output_shape)[2], (*q_shape)[2]) &&
         KnownDimsEqual((*output_shape)[3], (*q_shape)[3]);
}

// Absorb the exact UniRank rank-4 boundary layout when all four transposes
// are private to this attention core. The helper deliberately leaves the
// existing core-only fusion intact when the bundle is absent or unsupported.
void TryAppendBshdBoundaryTransposes(
    Ort::ConstNode score_matmul, Ort::ConstNode value_matmul,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::unordered_set<size_t>& selected_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  const auto score_inputs = score_matmul.GetInputs();
  const auto value_inputs = value_matmul.GetInputs();
  const auto value_outputs = value_matmul.GetOutputs();
  if (score_inputs.size() != 2 || value_inputs.size() != 2 ||
      value_outputs.size() != 1 ||
      graph_output_names.count(Name(value_outputs[0])) != 0) {
    return;
  }

  Ort::ConstNode q_transpose{nullptr};
  Ort::ConstNode k_transpose{nullptr};
  Ort::ConstNode v_transpose{nullptr};
  if (!GetProducer(score_inputs[0], q_transpose) ||
      !GetProducer(score_inputs[1], k_transpose) ||
      !GetProducer(value_inputs[1], v_transpose) ||
      !IsExactTranspose(q_transpose, {0, 2, 1, 3}) ||
      !IsExactTranspose(k_transpose, {0, 2, 3, 1}) ||
      !IsExactTranspose(v_transpose, {0, 2, 1, 3}) ||
      graph_output_names.count(Name(q_transpose.GetOutputs()[0])) != 0 ||
      graph_output_names.count(Name(k_transpose.GetOutputs()[0])) != 0 ||
      graph_output_names.count(Name(v_transpose.GetOutputs()[0])) != 0 ||
      !HasOnlyConsumer(q_transpose.GetOutputs()[0], score_matmul, 0) ||
      !HasOnlyConsumer(k_transpose.GetOutputs()[0], score_matmul, 1) ||
      !HasOnlyConsumer(v_transpose.GetOutputs()[0], value_matmul, 1)) {
    return;
  }

  const auto consumers = value_outputs[0].GetConsumers();
  if (consumers.size() != 1 || !IsOnnxOp(consumers[0].node, "Transpose") ||
      !IsExactTranspose(consumers[0].node, {0, 2, 1, 3}) ||
      consumers[0].index != 0 ||
      !BshdBoundaryShapesAreSupported(
          q_transpose.GetInputs()[0], k_transpose.GetInputs()[0],
          v_transpose.GetInputs()[0], consumers[0].node.GetOutputs()[0])) {
    return;
  }
  Ort::ConstNode output_transpose = consumers[0].node;
  if (output_transpose.GetOutputs().size() != 1) return;

  std::vector<Ort::ConstNode> candidate_nodes = {q_transpose, k_transpose,
                                                 v_transpose, output_transpose};
  std::vector<Ort::ConstNode> expanded_nodes = fusion_nodes;
  std::unordered_set<size_t> expanded_selected = selected_node_ids;
  for (Ort::ConstNode node : candidate_nodes) {
    if (!AddFusionNode(node, accepted_node_ids, expanded_selected,
                       expanded_nodes)) {
      return;
    }
  }
  if (!FusionHasNoExternalPathBetweenSelectedNodes(expanded_nodes,
                                                   expanded_selected)) {
    return;
  }
  fusion_nodes = std::move(expanded_nodes);
  selected_node_ids = std::move(expanded_selected);
}

bool IsCastTo(Ort::ConstNode node, ONNXTensorElementDataType elem_type) {
  return IsOnnxOp(node, "Cast") &&
         GetIntAttribute(node, "to").value_or(-1) == elem_type;
}

bool HasOnlyConsumers(Ort::ConstValueInfo value,
                      std::initializer_list<Ort::ConstNode> expected) {
  const auto consumers = value.GetConsumers();
  if (consumers.size() != expected.size()) {
    return false;
  }
  for (const auto& consumer : consumers) {
    bool found = false;
    for (Ort::ConstNode node : expected) {
      found |= consumer.node.GetId() == node.GetId();
    }
    if (!found) return false;
  }
  return true;
}

bool IsScalarFloatValue(Ort::ConstValueInfo value_info, float expected) {
  const auto value = ReadScalarFloatInitializer(value_info);
  if (!value.has_value()) return false;
  if (std::isinf(expected)) return std::isinf(*value) && *value < 0.0f;
  return *value == expected;
}

bool MhtaSdpaShapesHaveSameDataType(Ort::ConstValueInfo q,
                                    Ort::ConstValueInfo k,
                                    Ort::ConstValueInfo v,
                                    Ort::ConstValueInfo output) {
  return HasSameMhtaSdpaDataType(q, k) && HasSameMhtaSdpaDataType(q, v) &&
         HasSameMhtaSdpaDataType(q, output);
}

bool IsLastAxisSoftmax(Ort::ConstNode softmax_node,
                       Ort::ConstValueInfo softmax_input) {
  auto shape = GetTensorShape(softmax_input);
  if (!shape.has_value() || shape->empty()) {
    return false;
  }
  const int64_t rank = static_cast<int64_t>(shape->size());
  const int64_t default_axis = -1;
  int64_t axis = GetIntAttribute(softmax_node, "axis").value_or(default_axis);
  if (axis < 0) {
    axis += rank;
  }
  return axis == rank - 1;
}

std::optional<float> ScalarInputValue(Ort::ConstNode node, size_t data_index,
                                      size_t* scalar_index) {
  std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  if (inputs.size() != 2 || data_index > 1) {
    return std::nullopt;
  }
  const size_t candidate = 1 - data_index;
  auto value = ReadScalarFloatInitializer(inputs[candidate]);
  if (!value.has_value()) {
    return std::nullopt;
  }
  *scalar_index = candidate;
  return value;
}

bool HasConstantInitializerOtherInput(Ort::ConstNode node, size_t data_index,
                                      size_t* initializer_index) {
  std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  if (inputs.size() != 2 || data_index > 1) {
    return false;
  }
  const size_t candidate = 1 - data_index;
  if (!inputs[candidate].IsConstantInitializer()) {
    return false;
  }
  *initializer_index = candidate;
  return true;
}

bool ShapesAreSupported(Ort::ConstValueInfo q, Ort::ConstValueInfo k,
                        Ort::ConstValueInfo v, Ort::ConstValueInfo output) {
  // Keep the type gate local to this fusion: other matchers intentionally use
  // IsFloatTensorValueInfo() to mean FP32 only.
  if (!MhtaSdpaShapesHaveSameDataType(q, k, v, output)) {
    return false;
  }

  auto q_shape = GetTensorShape(q);
  auto k_shape = GetTensorShape(k);
  auto v_shape = GetTensorShape(v);
  auto output_shape = GetTensorShape(output);
  if (!q_shape.has_value() || !k_shape.has_value() || !v_shape.has_value() ||
      !output_shape.has_value() || q_shape->size() != 4 ||
      k_shape->size() != 4 || v_shape->size() != 4 ||
      output_shape->size() != 4) {
    return false;
  }

  // First implementation supports Q [B,H,S,D], K [B,H,D,S], V [B,H,S,D].
  return KnownDimsEqual((*q_shape)[0], (*k_shape)[0]) &&
         KnownDimsEqual((*q_shape)[0], (*v_shape)[0]) &&
         KnownDimsEqual((*q_shape)[1], (*k_shape)[1]) &&
         KnownDimsEqual((*q_shape)[1], (*v_shape)[1]) &&
         KnownDimsEqual((*q_shape)[2], (*k_shape)[3]) &&
         KnownDimsEqual((*q_shape)[2], (*v_shape)[2]) &&
         KnownDimsEqual((*q_shape)[3], (*k_shape)[2]) &&
         KnownDimsEqual((*q_shape)[3], (*v_shape)[3]) &&
         KnownDimsEqual((*output_shape)[0], (*q_shape)[0]) &&
         KnownDimsEqual((*output_shape)[1], (*q_shape)[1]) &&
         KnownDimsEqual((*output_shape)[2], (*q_shape)[2]) &&
         KnownDimsEqual((*output_shape)[3], (*v_shape)[3]);
}

bool IsUnsqueezeAxis2(Ort::ConstNode unsqueeze_node) {
  auto inputs = unsqueeze_node.GetInputs();
  if (inputs.size() != 2) {
    return false;
  }
  auto axis = ReadScalarIntInitializer(inputs[1]);
  return axis.has_value() && *axis == 2;
}

std::optional<int64_t> UnsqueezeAxis(Ort::ConstNode unsqueeze_node) {
  auto inputs = unsqueeze_node.GetInputs();
  if (inputs.size() != 2) {
    return std::nullopt;
  }
  return ReadScalarIntInitializer(inputs[1]);
}

bool ShapesAreSupportedForSimRank3(Ort::ConstValueInfo q, Ort::ConstValueInfo k,
                                   Ort::ConstValueInfo v,
                                   Ort::ConstValueInfo output) {
  if (!MhtaSdpaShapesHaveSameDataType(q, k, v, output)) {
    return false;
  }
  auto q_shape = GetTensorShape(q);
  auto k_shape = GetTensorShape(k);
  auto v_shape = GetTensorShape(v);
  auto output_shape = GetTensorShape(output);
  if ((q_shape.has_value() && q_shape->size() != 4) ||
      (k_shape.has_value() && k_shape->size() != 4) ||
      (v_shape.has_value() && v_shape->size() != 4) ||
      (output_shape.has_value() && output_shape->size() != 3)) {
    return false;
  }

  if (q_shape.has_value() && !KnownDimsEqual((*q_shape)[1], 1)) {
    return false;
  }
  if (q_shape.has_value() && output_shape.has_value() &&
      (!KnownDimsEqual((*q_shape)[0], (*output_shape)[0]) ||
       !KnownDimsEqual((*output_shape)[1], (*q_shape)[1]))) {
    return false;
  }
  if (q_shape.has_value() && k_shape.has_value() &&
      (!KnownDimsEqual((*k_shape)[2], (*q_shape)[2]) ||
       !KnownDimsEqual((*k_shape)[3], (*q_shape)[3]))) {
    return false;
  }
  if (q_shape.has_value() && v_shape.has_value() &&
      (!KnownDimsEqual((*v_shape)[1], (*q_shape)[2]) ||
       !KnownDimsEqual((*v_shape)[3], (*q_shape)[3]))) {
    return false;
  }
  if (k_shape.has_value() && v_shape.has_value() &&
      !KnownDimsEqual((*k_shape)[1], (*v_shape)[2])) {
    return false;
  }
  if (q_shape.has_value() && output_shape.has_value()) {
    const int64_t heads = (*q_shape)[2];
    const int64_t head_dim = (*q_shape)[3];
    const int64_t flat_dim = (*output_shape)[2];
    if (heads > 0 && head_dim > 0 && flat_dim > 0 &&
        flat_dim != heads * head_dim) {
      return false;
    }
  }

  return true;
}

bool ShapesAreSupportedForFoldedHeadRank3(Ort::ConstValueInfo q,
                                          Ort::ConstValueInfo k,
                                          Ort::ConstValueInfo v,
                                          Ort::ConstValueInfo output) {
  if (!IsFloat32TensorValueInfo(q) || !HasSameMhtaSdpaDataType(q, k) ||
      !HasSameMhtaSdpaDataType(q, v) || !HasSameMhtaSdpaDataType(q, output)) {
    return false;
  }
  auto q_shape = GetTensorShape(q);
  auto k_shape = GetTensorShape(k);
  auto v_shape = GetTensorShape(v);
  auto output_shape = GetTensorShape(output);
  if (!q_shape.has_value() || !k_shape.has_value() || !v_shape.has_value() ||
      !output_shape.has_value() || q_shape->size() != 3 ||
      k_shape->size() != 3 || v_shape->size() != 3 ||
      output_shape->size() != 3) {
    return false;
  }
  return KnownDimsEqual((*q_shape)[0], (*k_shape)[0]) &&
         KnownDimsEqual((*q_shape)[0], (*v_shape)[0]) &&
         KnownDimsEqual((*q_shape)[2], (*k_shape)[1]) &&
         KnownDimsEqual((*q_shape)[2], (*v_shape)[2]) &&
         KnownDimsEqual((*k_shape)[2], (*v_shape)[1]) &&
         KnownDimsEqual((*output_shape)[0], (*q_shape)[0]) &&
         KnownDimsEqual((*output_shape)[1], (*q_shape)[1]) &&
         KnownDimsEqual((*output_shape)[2], (*v_shape)[2]);
}

bool CanFuseMhtaScaledDotProductAttention(
    Ort::ConstNode value_matmul,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(value_matmul, "MatMul") ||
      accepted_node_ids.count(value_matmul.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> value_matmul_inputs =
      value_matmul.GetInputs();
  std::vector<Ort::ConstValueInfo> value_matmul_outputs =
      value_matmul.GetOutputs();
  if (value_matmul_inputs.size() != 2 || value_matmul_outputs.size() != 1) {
    return false;
  }

  Ort::ConstNode softmax_node{nullptr};
  if (!GetProducer(value_matmul_inputs[0], softmax_node) ||
      !IsOnnxOp(softmax_node, "Softmax") ||
      accepted_node_ids.count(softmax_node.GetId()) != 0) {
    return false;
  }
  if (!HasSingleConsumer(value_matmul_inputs[0], graph_output_names)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> softmax_inputs = softmax_node.GetInputs();
  std::vector<Ort::ConstValueInfo> softmax_outputs = softmax_node.GetOutputs();
  if (softmax_inputs.size() != 1 || softmax_outputs.size() != 1 ||
      Name(softmax_outputs[0]) != Name(value_matmul_inputs[0]) ||
      !IsLastAxisSoftmax(softmax_node, softmax_inputs[0])) {
    return false;
  }

  Ort::ConstNode div_node{nullptr};
  if (!GetProducer(softmax_inputs[0], div_node) || !IsOnnxOp(div_node, "Div") ||
      accepted_node_ids.count(div_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> div_inputs = div_node.GetInputs();
  std::vector<Ort::ConstValueInfo> div_outputs = div_node.GetOutputs();
  if (div_inputs.size() != 2 || div_outputs.size() != 1 ||
      Name(div_outputs[0]) != Name(softmax_inputs[0]) ||
      !HasSingleConsumer(div_outputs[0], graph_output_names)) {
    return false;
  }
  size_t temperature_index = 0;
  auto temperature = ScalarInputValue(div_node, 0, &temperature_index);
  if (!temperature.has_value() || *temperature == 0.0f ||
      temperature_index != 1) {
    return false;
  }

  Ort::ConstNode add_node{nullptr};
  if (!GetProducer(div_inputs[0], add_node) || !IsOnnxOp(add_node, "Add") ||
      accepted_node_ids.count(add_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  if (add_inputs.size() != 2 || add_outputs.size() != 1 ||
      Name(add_outputs[0]) != Name(div_inputs[0]) ||
      !HasSingleConsumer(add_outputs[0], graph_output_names)) {
    return false;
  }

  Ort::ConstNode mul_node{nullptr};
  int64_t mul_input_index = -1;
  for (int64_t i = 0; i < 2; ++i) {
    Ort::ConstNode producer{nullptr};
    if (GetProducer(add_inputs[static_cast<size_t>(i)], producer) &&
        IsOnnxOp(producer, "Mul")) {
      mul_node = producer;
      mul_input_index = i;
      break;
    }
  }
  if (!mul_node || accepted_node_ids.count(mul_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> mul_inputs = mul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> mul_outputs = mul_node.GetOutputs();
  if (mul_inputs.size() != 2 || mul_outputs.size() != 1 ||
      Name(mul_outputs[0]) !=
          Name(add_inputs[static_cast<size_t>(mul_input_index)]) ||
      !HasSingleConsumer(mul_outputs[0], graph_output_names)) {
    return false;
  }
  size_t scale_index = 0;
  auto scale = ScalarInputValue(mul_node, 0, &scale_index);
  if (!scale.has_value() || scale_index != 1) {
    return false;
  }

  Ort::ConstNode score_matmul{nullptr};
  if (!GetProducer(mul_inputs[0], score_matmul) ||
      !IsOnnxOp(score_matmul, "MatMul") ||
      accepted_node_ids.count(score_matmul.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> score_inputs = score_matmul.GetInputs();
  std::vector<Ort::ConstValueInfo> score_outputs = score_matmul.GetOutputs();
  if (score_inputs.size() != 2 || score_outputs.size() != 1 ||
      Name(score_outputs[0]) != Name(mul_inputs[0]) ||
      !HasSingleConsumer(score_outputs[0], graph_output_names)) {
    return false;
  }

  const size_t mask_input_index = static_cast<size_t>(1 - mul_input_index);
  if (!IsZeroFloatInitializer(add_inputs[mask_input_index])) {
    return false;
  }

  if (!ShapesAreSupported(score_inputs[0], score_inputs[1],
                          value_matmul_inputs[1], value_matmul_outputs[0])) {
    return false;
  }

  fusion_nodes = {score_matmul, mul_node,     add_node,
                  div_node,     softmax_node, value_matmul};
  std::unordered_set<size_t> selected;
  for (Ort::ConstNode node : fusion_nodes) selected.insert(node.GetId());
  TryAppendBshdBoundaryTransposes(score_matmul, value_matmul,
                                  graph_output_names, accepted_node_ids,
                                  selected, fusion_nodes);
  if (!FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes, selected)) {
    return false;
  }
  return true;
}

// Matches the boolean keep-mask form emitted by ranking-gr:
//   MatMul -> Mul(scale) -> Where(mask, score, -inf) -> Softmax
//   -> Where(mask, probability, 0) -> MatMul(V)
bool TryAppendRawInt32MaskPreprocessing(
    Ort::ConstValueInfo boolean_mask, Ort::ConstNode pre_where,
    Ort::ConstNode post_where,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::unordered_set<size_t>& selected,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  Ort::ConstNode equal{nullptr};
  if (!GetProducer(boolean_mask, equal) || !IsOnnxOp(equal, "Equal")) {
    return true;
  }
  const auto equal_inputs = equal.GetInputs();
  const auto equal_outputs = equal.GetOutputs();
  if (equal_inputs.size() != 2 || equal_outputs.size() != 1 ||
      graph_output_names.count(Name(equal_outputs[0])) != 0) {
    return false;
  }
  Ort::ConstNode cast{nullptr};
  bool has_one = false;
  for (Ort::ConstValueInfo input : equal_inputs) {
    Ort::ConstNode producer{nullptr};
    if (GetProducer(input, producer) &&
        IsCastTo(producer, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
      cast = producer;
    } else {
      has_one |= ReadScalarIntInitializer(input).value_or(0) == 1;
    }
  }
  if (!cast || !has_one || !IsBoolTensorValueInfo(equal_outputs[0]) ||
      !HasOnlyConsumers(equal_outputs[0], {pre_where, post_where})) {
    return false;
  }

  const auto cast_inputs = cast.GetInputs();
  const auto cast_outputs = cast.GetOutputs();
  Ort::ConstNode slice{nullptr};
  if (cast_inputs.size() != 1 || cast_outputs.size() != 1 ||
      graph_output_names.count(Name(cast_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(cast_outputs[0], equal) ||
      !GetProducer(cast_inputs[0], slice) || !IsOnnxOp(slice, "Slice")) {
    return false;
  }

  const auto slice_inputs = slice.GetInputs();
  const auto slice_outputs = slice.GetOutputs();
  const auto starts = slice_inputs.size() == 5
                          ? ReadIntInitializerNoLimit(slice_inputs[1])
                          : std::nullopt;
  const auto axes = slice_inputs.size() == 5
                        ? ReadIntInitializerNoLimit(slice_inputs[3])
                        : std::nullopt;
  const auto steps = slice_inputs.size() == 5
                         ? ReadIntInitializerNoLimit(slice_inputs[4])
                         : std::nullopt;
  if (slice_inputs.size() != 5 || slice_outputs.size() != 1) return false;
  if (!IsInt32TensorValueInfo(slice_inputs[0]) ||
      !IsInt32TensorValueInfo(slice_outputs[0]))
    return false;
  if (!IsInt64TensorValueInfo(slice_inputs[2])) return false;
  if (graph_output_names.count(Name(slice_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(slice_outputs[0], cast))
    return false;
  if (!starts.has_value() || *starts != std::vector<int64_t>{0}) return false;
  if (!axes.has_value() || *axes != std::vector<int64_t>{3}) return false;
  if (!steps.has_value() || *steps != std::vector<int64_t>{1}) return false;

  for (Ort::ConstNode node : {slice, cast, equal}) {
    if (!AddFusionNode(node, accepted_node_ids, selected, fusion_nodes)) {
      return false;
    }
  }
  return true;
}

bool ResolvePostWhereProbabilityInput(
    Ort::ConstValueInfo post_probability_input,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    Ort::ConstNode& softmax, std::vector<Ort::ConstNode>& post_softmax_casts) {
  Ort::ConstNode producer{nullptr};
  if (!GetProducer(post_probability_input, producer)) return false;
  if (IsOnnxOp(producer, "Softmax")) {
    softmax = producer;
    return accepted_node_ids.count(softmax.GetId()) == 0;
  }
  if (!IsOnnxOp(producer, "Cast") ||
      accepted_node_ids.count(producer.GetId()) != 0) {
    return false;
  }
  Ort::ConstNode cast_to_storage = producer;
  const auto cast_to_storage_inputs = cast_to_storage.GetInputs();
  const auto cast_to_storage_outputs = cast_to_storage.GetOutputs();
  if (cast_to_storage_inputs.size() != 1 ||
      cast_to_storage_outputs.size() != 1 ||
      !HasSingleConsumer(cast_to_storage_outputs[0], graph_output_names)) {
    return false;
  }
  const ONNXTensorElementDataType storage_type =
      post_probability_input.TypeInfo()
          .GetTensorTypeAndShapeInfo()
          .GetElementType();
  if (!IsCastTo(cast_to_storage, storage_type)) return false;

  Ort::ConstNode cast_to_float{nullptr};
  if (!GetProducer(cast_to_storage_inputs[0], cast_to_float) ||
      !IsOnnxOp(cast_to_float, "Cast") ||
      accepted_node_ids.count(cast_to_float.GetId()) != 0 ||
      !IsCastTo(cast_to_float, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    return false;
  }
  const auto cast_to_float_inputs = cast_to_float.GetInputs();
  const auto cast_to_float_outputs = cast_to_float.GetOutputs();
  if (cast_to_float_inputs.size() != 1 || cast_to_float_outputs.size() != 1 ||
      Name(cast_to_float_outputs[0]) != Name(cast_to_storage_inputs[0]) ||
      !HasSingleConsumer(cast_to_float_outputs[0], graph_output_names)) {
    return false;
  }
  if (!GetProducer(cast_to_float_inputs[0], softmax) ||
      !IsOnnxOp(softmax, "Softmax") ||
      accepted_node_ids.count(softmax.GetId()) != 0) {
    return false;
  }
  post_softmax_casts = {cast_to_float, cast_to_storage};
  return true;
}

bool CanFuseBooleanMhtaScaledDotProductAttention(
    Ort::ConstNode value_matmul,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(value_matmul, "MatMul") ||
      accepted_node_ids.count(value_matmul.GetId()) != 0)
    return false;
  const auto value_inputs = value_matmul.GetInputs();
  const auto value_outputs = value_matmul.GetOutputs();
  if (value_inputs.size() != 2 || value_outputs.size() != 1) return false;

  Ort::ConstNode post_where{nullptr};
  if (!GetProducer(value_inputs[0], post_where) ||
      !IsOnnxOp(post_where, "Where") ||
      accepted_node_ids.count(post_where.GetId()) != 0)
    return false;
  const auto post_inputs = post_where.GetInputs();
  const auto post_outputs = post_where.GetOutputs();
  if (post_inputs.size() != 3 || post_outputs.size() != 1 ||
      !HasSingleConsumer(post_outputs[0], graph_output_names) ||
      Name(post_outputs[0]) != Name(value_inputs[0]) ||
      !IsBoolTensorValueInfo(post_inputs[0]) ||
      !IsScalarFloatValue(post_inputs[2], 0.0f))
    return false;

  Ort::ConstNode softmax{nullptr};
  std::vector<Ort::ConstNode> post_softmax_casts;
  if (!ResolvePostWhereProbabilityInput(post_inputs[1], graph_output_names,
                                        accepted_node_ids, softmax,
                                        post_softmax_casts))
    return false;
  const auto softmax_inputs = softmax.GetInputs();
  const auto softmax_outputs = softmax.GetOutputs();
  if (softmax_inputs.size() != 1 || softmax_outputs.size() != 1 ||
      !HasSingleConsumer(softmax_outputs[0], graph_output_names) ||
      !IsLastAxisSoftmax(softmax, softmax_inputs[0]))
    return false;

  Ort::ConstNode pre_where{nullptr};
  if (!GetProducer(softmax_inputs[0], pre_where) ||
      !IsOnnxOp(pre_where, "Where") ||
      accepted_node_ids.count(pre_where.GetId()) != 0)
    return false;
  const auto pre_inputs = pre_where.GetInputs();
  const auto pre_outputs = pre_where.GetOutputs();
  if (pre_inputs.size() != 3 || pre_outputs.size() != 1 ||
      Name(pre_outputs[0]) != Name(softmax_inputs[0]) ||
      !HasSingleConsumer(pre_outputs[0], graph_output_names) ||
      !IsBoolTensorValueInfo(pre_inputs[0]) ||
      Name(pre_inputs[0]) != Name(post_inputs[0]) ||
      !IsScalarFloatValue(pre_inputs[2], -INFINITY))
    return false;

  Ort::ConstNode scale_mul{nullptr};
  if (!GetProducer(pre_inputs[1], scale_mul) || !IsOnnxOp(scale_mul, "Mul") ||
      accepted_node_ids.count(scale_mul.GetId()) != 0)
    return false;
  const auto scale_inputs = scale_mul.GetInputs();
  const auto scale_outputs = scale_mul.GetOutputs();
  if (scale_inputs.size() != 2 || scale_outputs.size() != 1 ||
      Name(scale_outputs[0]) != Name(pre_inputs[1]) ||
      !HasSingleConsumer(scale_outputs[0], graph_output_names))
    return false;
  size_t scale_index = 0;
  const auto scale = ScalarInputValue(scale_mul, 0, &scale_index);
  if (!scale.has_value() || scale_index != 1 || !std::isfinite(*scale))
    return false;

  Ort::ConstNode score_matmul{nullptr};
  if (!GetProducer(scale_inputs[0], score_matmul) ||
      !IsOnnxOp(score_matmul, "MatMul") ||
      accepted_node_ids.count(score_matmul.GetId()) != 0)
    return false;
  const auto score_inputs = score_matmul.GetInputs();
  const auto score_outputs = score_matmul.GetOutputs();
  if (score_inputs.size() != 2 || score_outputs.size() != 1 ||
      Name(score_outputs[0]) != Name(scale_inputs[0]) ||
      !HasSingleConsumer(score_outputs[0], graph_output_names) ||
      !ShapesAreSupported(score_inputs[0], score_inputs[1], value_inputs[1],
                          value_outputs[0]))
    return false;

  std::unordered_set<size_t> selected;
  for (Ort::ConstNode node : {score_matmul, scale_mul, pre_where, softmax}) {
    if (!AddFusionNode(node, accepted_node_ids, selected, fusion_nodes))
      return false;
  }
  for (Ort::ConstNode node : post_softmax_casts) {
    if (!AddFusionNode(node, accepted_node_ids, selected, fusion_nodes))
      return false;
  }
  for (Ort::ConstNode node : {post_where, value_matmul}) {
    if (!AddFusionNode(node, accepted_node_ids, selected, fusion_nodes))
      return false;
  }
  if (!TryAppendRawInt32MaskPreprocessing(pre_inputs[0], pre_where, post_where,
                                          graph_output_names, accepted_node_ids,
                                          selected, fusion_nodes)) {
    return false;
  }
  TryAppendBshdBoundaryTransposes(score_matmul, value_matmul,
                                  graph_output_names, accepted_node_ids,
                                  selected, fusion_nodes);
  return FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes, selected);
}

bool CanFuseLseqPreMaskMhtaScaledDotProductAttention(
    Ort::ConstNode value_matmul,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(value_matmul, "MatMul") ||
      accepted_node_ids.count(value_matmul.GetId()) != 0)
    return false;
  const auto value_inputs = value_matmul.GetInputs();
  const auto value_outputs = value_matmul.GetOutputs();
  if (value_inputs.size() != 2 || value_outputs.size() != 1) return false;

  Ort::ConstNode softmax{nullptr};
  if (!GetProducer(value_inputs[0], softmax) || !IsOnnxOp(softmax, "Softmax") ||
      accepted_node_ids.count(softmax.GetId()) != 0)
    return false;
  const auto softmax_inputs = softmax.GetInputs();
  const auto softmax_outputs = softmax.GetOutputs();
  if (softmax_inputs.size() != 1 || softmax_outputs.size() != 1 ||
      Name(softmax_outputs[0]) != Name(value_inputs[0]) ||
      !HasSingleConsumer(softmax_outputs[0], graph_output_names) ||
      !IsLastAxisSoftmax(softmax, softmax_inputs[0]))
    return false;

  Ort::ConstNode pre_where{nullptr};
  if (!GetProducer(softmax_inputs[0], pre_where) ||
      !IsOnnxOp(pre_where, "Where") ||
      accepted_node_ids.count(pre_where.GetId()) != 0)
    return false;
  const auto where_inputs = pre_where.GetInputs();
  const auto where_outputs = pre_where.GetOutputs();
  if (where_inputs.size() != 3 || where_outputs.size() != 1 ||
      Name(where_outputs[0]) != Name(softmax_inputs[0]) ||
      !HasSingleConsumer(where_outputs[0], graph_output_names) ||
      !IsBoolTensorValueInfo(where_inputs[0]) ||
      !IsScalarFloatValue(where_inputs[2], -INFINITY))
    return false;

  Ort::ConstNode scale_mul{nullptr};
  if (!GetProducer(where_inputs[1], scale_mul) || !IsOnnxOp(scale_mul, "Mul") ||
      accepted_node_ids.count(scale_mul.GetId()) != 0)
    return false;
  const auto scale_inputs = scale_mul.GetInputs();
  const auto scale_outputs = scale_mul.GetOutputs();
  if (scale_inputs.size() != 2 || scale_outputs.size() != 1 ||
      Name(scale_outputs[0]) != Name(where_inputs[1]))
    return false;
  size_t scale_index = 0;
  const auto scale = ScalarInputValue(scale_mul, 0, &scale_index);
  if (!scale.has_value() || scale_index != 1 || !std::isfinite(*scale))
    return false;

  Ort::ConstNode score_matmul{nullptr};
  if (!GetProducer(scale_inputs[0], score_matmul) ||
      !IsOnnxOp(score_matmul, "MatMul") ||
      accepted_node_ids.count(score_matmul.GetId()) != 0)
    return false;
  const auto score_inputs = score_matmul.GetInputs();
  const auto score_outputs = score_matmul.GetOutputs();
  if (score_inputs.size() != 2 || score_outputs.size() != 1 ||
      Name(score_outputs[0]) != Name(scale_inputs[0]) ||
      !HasSingleConsumer(score_outputs[0], graph_output_names) ||
      !ShapesAreSupportedForFoldedHeadRank3(score_inputs[0], score_inputs[1],
                                            value_inputs[1], value_outputs[0]))
    return false;

  Ort::ConstNode unsqueeze_1{nullptr};
  Ort::ConstNode unsqueeze_0{nullptr};
  if (!GetProducer(where_inputs[0], unsqueeze_1) ||
      !IsOnnxOp(unsqueeze_1, "Unsqueeze") ||
      accepted_node_ids.count(unsqueeze_1.GetId()) != 0)
    return false;
  const auto unsqueeze_1_inputs = unsqueeze_1.GetInputs();
  const auto unsqueeze_1_outputs = unsqueeze_1.GetOutputs();
  const auto axis_1 = UnsqueezeAxis(unsqueeze_1);
  if (unsqueeze_1_inputs.size() != 2 || unsqueeze_1_outputs.size() != 1 ||
      Name(unsqueeze_1_outputs[0]) != Name(where_inputs[0]) ||
      !HasSingleConsumer(unsqueeze_1_outputs[0], graph_output_names) ||
      !axis_1.has_value() || *axis_1 != 1 ||
      !GetProducer(unsqueeze_1_inputs[0], unsqueeze_0) ||
      !IsOnnxOp(unsqueeze_0, "Unsqueeze") ||
      accepted_node_ids.count(unsqueeze_0.GetId()) != 0)
    return false;
  const auto unsqueeze_0_inputs = unsqueeze_0.GetInputs();
  const auto unsqueeze_0_outputs = unsqueeze_0.GetOutputs();
  const auto axis_0 = UnsqueezeAxis(unsqueeze_0);
  if (unsqueeze_0_inputs.size() != 2 || unsqueeze_0_outputs.size() != 1 ||
      Name(unsqueeze_0_outputs[0]) != Name(unsqueeze_1_inputs[0]) ||
      !HasSingleConsumer(unsqueeze_0_outputs[0], graph_output_names) ||
      !axis_0.has_value() || *axis_0 != 0)
    return false;

  Ort::ConstNode less{nullptr};
  if (!GetProducer(unsqueeze_0_inputs[0], less) || !IsOnnxOp(less, "Less") ||
      accepted_node_ids.count(less.GetId()) != 0)
    return false;
  const auto less_inputs = less.GetInputs();
  const auto less_outputs = less.GetOutputs();
  if (less_inputs.size() != 2 || less_outputs.size() != 1 ||
      Name(less_outputs[0]) != Name(unsqueeze_0_inputs[0]) ||
      !HasSingleConsumer(less_outputs[0], graph_output_names))
    return false;

  Ort::ConstNode range{nullptr};
  Ort::ConstNode clip{nullptr};
  if (!GetProducer(less_inputs[0], range) || !IsOnnxOp(range, "Range") ||
      !GetProducer(less_inputs[1], clip) || !IsOnnxOp(clip, "Clip") ||
      accepted_node_ids.count(range.GetId()) != 0 ||
      accepted_node_ids.count(clip.GetId()) != 0)
    return false;
  const auto range_inputs = range.GetInputs();
  const auto range_outputs = range.GetOutputs();
  if (range_inputs.size() != 3 || range_outputs.size() != 1 ||
      Name(range_outputs[0]) != Name(less_inputs[0]) ||
      !HasSingleConsumer(range_outputs[0], graph_output_names) ||
      ReadScalarIntInitializer(range_inputs[0]).value_or(-1) != 0 ||
      ReadScalarIntInitializer(range_inputs[2]).value_or(-1) != 1)
    return false;

  const auto clip_inputs = clip.GetInputs();
  const auto clip_outputs = clip.GetOutputs();
  if (clip_inputs.size() < 2 || clip_outputs.size() != 1 ||
      Name(clip_outputs[0]) != Name(less_inputs[1]) ||
      !HasSingleConsumer(clip_outputs[0], graph_output_names) ||
      ReadScalarIntInitializer(clip_inputs[1]).value_or(-1) != 1)
    return false;

  Ort::ConstNode sub{nullptr};
  if (!GetProducer(clip_inputs[0], sub) || !IsOnnxOp(sub, "Sub") ||
      accepted_node_ids.count(sub.GetId()) != 0)
    return false;
  const auto sub_inputs = sub.GetInputs();
  const auto sub_outputs = sub.GetOutputs();
  if (sub_inputs.size() != 2 || sub_outputs.size() != 1 ||
      Name(sub_outputs[0]) != Name(clip_inputs[0]) ||
      !HasSingleConsumer(sub_outputs[0], graph_output_names) ||
      ReadScalarIntInitializer(sub_inputs[1]).value_or(-1) != 1)
    return false;

  Ort::ConstNode gather{nullptr};
  if (!GetProducer(sub_inputs[0], gather) || !IsOnnxOp(gather, "Gather") ||
      accepted_node_ids.count(gather.GetId()) != 0 ||
      Name(range_inputs[1]) != Name(sub_inputs[0]))
    return false;
  const auto gather_inputs = gather.GetInputs();
  const auto gather_outputs = gather.GetOutputs();
  if (gather_inputs.size() != 2 || gather_outputs.size() != 1 ||
      !HasOnlyConsumers(gather_outputs[0], {sub, range}) ||
      GetIntAttribute(gather, "axis").value_or(0) != 0 ||
      ReadScalarIntInitializer(gather_inputs[1]).value_or(-1) != 2)
    return false;

  Ort::ConstNode shape{nullptr};
  if (!GetProducer(gather_inputs[0], shape) || !IsOnnxOp(shape, "Shape") ||
      accepted_node_ids.count(shape.GetId()) != 0)
    return false;
  const auto shape_inputs = shape.GetInputs();
  const auto shape_outputs = shape.GetOutputs();
  if (shape_inputs.size() != 1 || shape_outputs.size() != 1 ||
      Name(shape_inputs[0]) != Name(scale_outputs[0]) ||
      Name(shape_outputs[0]) != Name(gather_inputs[0]) ||
      !HasSingleConsumer(shape_outputs[0], graph_output_names) ||
      !HasOnlyConsumers(scale_outputs[0], {shape, pre_where}))
    return false;

  std::unordered_set<size_t> selected;
  for (Ort::ConstNode node :
       {score_matmul, scale_mul, shape, gather, sub, clip, range, less,
        unsqueeze_0, unsqueeze_1, pre_where, softmax, value_matmul}) {
    if (!AddFusionNode(node, accepted_node_ids, selected, fusion_nodes))
      return false;
  }
  if (!FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes, selected))
    return false;

  // The optimized UniRank graph keeps the Q/K/V tensors in BF16 at the
  // fusion boundary, then widens them to FP32 only for the legacy rank-3
  // attention core.  Absorb that private widen/transpose/zero-padding path
  // when it is fully self-contained.  The runtime can then expose the raw
  // [S,H,D] BF16 tensors directly to FlashAttention instead of materializing
  // FP32 copies.
  {
    Ort::ConstNode q_transpose{nullptr};
    Ort::ConstNode k_transpose{nullptr};
    Ort::ConstNode v_transpose{nullptr};
    Ort::ConstNode output_transpose{nullptr};
    Ort::ConstNode q_cast{nullptr};
    Ort::ConstNode k_cast{nullptr};
    Ort::ConstNode v_cast{nullptr};
    Ort::ConstNode output_cast{nullptr};
    Ort::ConstNode k_concat{nullptr};
    Ort::ConstNode v_concat{nullptr};
    Ort::ConstNode zero_node{nullptr};

    const auto score_inputs = score_matmul.GetInputs();
    const auto value_inputs = value_matmul.GetInputs();
    const auto value_outputs = value_matmul.GetOutputs();
    if (score_inputs.size() == 2 && value_inputs.size() == 2 &&
        value_outputs.size() == 1 &&
        GetProducer(score_inputs[0], q_transpose) &&
        GetProducer(score_inputs[1], k_transpose) &&
        GetProducer(value_inputs[1], v_transpose) &&
        IsExactTranspose(q_transpose, {1, 0, 2}) &&
        IsExactTranspose(k_transpose, {1, 2, 0}) &&
        IsExactTranspose(v_transpose, {1, 0, 2}) &&
        HasOnlyConsumer(q_transpose.GetOutputs()[0], score_matmul, 0) &&
        HasOnlyConsumer(k_transpose.GetOutputs()[0], score_matmul, 1) &&
        HasOnlyConsumer(v_transpose.GetOutputs()[0], value_matmul, 1) &&
        GetProducer(q_transpose.GetInputs()[0], q_cast) &&
        GetProducer(k_transpose.GetInputs()[0], k_concat) &&
        GetProducer(v_transpose.GetInputs()[0], v_concat) &&
        IsCastTo(q_cast, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) &&
        IsOnnxOp(k_concat, "Concat") && IsOnnxOp(v_concat, "Concat")) {
      const auto q_cast_inputs = q_cast.GetInputs();
      const auto q_cast_outputs = q_cast.GetOutputs();
      const auto k_concat_inputs = k_concat.GetInputs();
      const auto v_concat_inputs = v_concat.GetInputs();
      const auto k_concat_outputs = k_concat.GetOutputs();
      const auto v_concat_outputs = v_concat.GetOutputs();
      auto q_shape = q_cast_inputs.size() == 1
                         ? GetTensorShape(q_cast_inputs[0])
                         : std::nullopt;
      auto k_shape = std::optional<std::vector<int64_t>>{};
      auto v_shape = std::optional<std::vector<int64_t>>{};
      bool boundary_ok =
          q_cast_inputs.size() == 1 && q_cast_outputs.size() == 1 &&
          IsBfloat16TensorValueInfo(q_cast_inputs[0]) && q_shape.has_value() &&
          q_shape->size() == 3 && k_concat_inputs.size() == 2 &&
          v_concat_inputs.size() == 2 && k_concat_outputs.size() == 1 &&
          v_concat_outputs.size() == 1 &&
          GetIntAttribute(k_concat, "axis").value_or(-1) == 0 &&
          GetIntAttribute(v_concat, "axis").value_or(-1) == 0 &&
          GetProducer(k_concat_inputs[0], k_cast) &&
          GetProducer(v_concat_inputs[0], v_cast) &&
          IsCastTo(k_cast, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) &&
          IsCastTo(v_cast, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) &&
          k_cast.GetInputs().size() == 1 && v_cast.GetInputs().size() == 1 &&
          IsBfloat16TensorValueInfo(k_cast.GetInputs()[0]) &&
          IsBfloat16TensorValueInfo(v_cast.GetInputs()[0]);
      if (boundary_ok) {
        k_shape = GetTensorShape(k_cast.GetInputs()[0]);
        v_shape = GetTensorShape(v_cast.GetInputs()[0]);
        boundary_ok = k_shape.has_value() && v_shape.has_value() &&
                      k_shape->size() == 3 && v_shape->size() == 3 &&
                      KnownDimsEqual((*q_shape)[1], (*k_shape)[1]) &&
                      KnownDimsEqual((*q_shape)[1], (*v_shape)[1]) &&
                      KnownDimsEqual((*q_shape)[2], (*k_shape)[2]) &&
                      KnownDimsEqual((*q_shape)[2], (*v_shape)[2]) &&
                      KnownDimsEqual((*k_shape)[0], (*v_shape)[0]) &&
                      GetProducer(k_concat_inputs[1], zero_node);
      }
      if (boundary_ok) {
        Ort::ConstNode v_zero{nullptr};
        boundary_ok =
            GetProducer(v_concat_inputs[1], v_zero) &&
            v_zero.GetId() == zero_node.GetId() &&
            IsZeroFloatConstantOfShape(zero_node) &&
            HasOnlyConsumers(zero_node.GetOutputs()[0], {k_concat, v_concat}) &&
            HasOnlyConsumer(k_concat_outputs[0], k_transpose, 0) &&
            HasOnlyConsumer(v_concat_outputs[0], v_transpose, 0);
      }
      if (boundary_ok) {
        const auto output_consumers = value_outputs[0].GetConsumers();
        if (output_consumers.size() != 1 || output_consumers[0].index != 0) {
          boundary_ok = false;
        } else {
          output_transpose = output_consumers[0].node;
        }
        if (boundary_ok && (!IsExactTranspose(output_transpose, {1, 0, 2}) ||
                            output_transpose.GetOutputs().size() != 1)) {
          boundary_ok = false;
        }
      }
      if (boundary_ok) {
        const auto output_cast_consumers =
            output_transpose.GetOutputs()[0].GetConsumers();
        if (output_cast_consumers.size() != 1 ||
            output_cast_consumers[0].index != 0) {
          boundary_ok = false;
        } else {
          output_cast = output_cast_consumers[0].node;
        }
        if (boundary_ok &&
            (!IsCastTo(output_cast, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) ||
             output_cast.GetInputs().size() != 1 ||
             output_cast.GetOutputs().size() != 1 ||
             !IsFloat32TensorValueInfo(output_cast.GetInputs()[0]) ||
             !IsBfloat16TensorValueInfo(output_cast.GetOutputs()[0]))) {
          boundary_ok = false;
        }
      }
      if (boundary_ok) {
        std::vector<Ort::ConstNode> expanded = fusion_nodes;
        auto add = [&](Ort::ConstNode node) {
          return AddFusionNode(node, accepted_node_ids, selected, expanded);
        };
        for (Ort::ConstNode node :
             {q_cast, q_transpose, k_cast, k_concat, k_transpose, v_cast,
              v_concat, v_transpose, output_transpose, output_cast}) {
          if (!add(node)) {
            boundary_ok = false;
            break;
          }
        }
        if (boundary_ok &&
            FusionHasNoExternalPathBetweenSelectedNodes(expanded, selected)) {
          fusion_nodes = std::move(expanded);
        }
      }
    }
  }
  return true;
}

bool CanFuseSimRank3MhtaScaledDotProductAttention(
    Ort::ConstNode output_reshape,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(output_reshape, "Reshape") ||
      accepted_node_ids.count(output_reshape.GetId()) != 0) {
    return false;
  }
  auto reshape_inputs = output_reshape.GetInputs();
  auto reshape_outputs = output_reshape.GetOutputs();
  if (reshape_inputs.size() != 2 || reshape_outputs.size() != 1) {
    return false;
  }

  Ort::ConstNode value_matmul{nullptr};
  if (!GetProducer(reshape_inputs[0], value_matmul) ||
      !IsOnnxOp(value_matmul, "MatMul") ||
      accepted_node_ids.count(value_matmul.GetId()) != 0) {
    return false;
  }
  auto value_inputs = value_matmul.GetInputs();
  auto value_outputs = value_matmul.GetOutputs();
  if (value_inputs.size() != 2 || value_outputs.size() != 1 ||
      Name(value_outputs[0]) != Name(reshape_inputs[0]) ||
      !HasSingleConsumer(value_outputs[0], graph_output_names)) {
    return false;
  }

  Ort::ConstNode unsqueeze_node{nullptr};
  if (!GetProducer(value_inputs[0], unsqueeze_node) ||
      !IsOnnxOp(unsqueeze_node, "Unsqueeze") ||
      accepted_node_ids.count(unsqueeze_node.GetId()) != 0 ||
      !IsUnsqueezeAxis2(unsqueeze_node)) {
    return false;
  }
  auto unsqueeze_inputs = unsqueeze_node.GetInputs();
  auto unsqueeze_outputs = unsqueeze_node.GetOutputs();
  if (unsqueeze_inputs.empty() || unsqueeze_outputs.size() != 1 ||
      Name(unsqueeze_outputs[0]) != Name(value_inputs[0]) ||
      !HasSingleConsumer(unsqueeze_outputs[0], graph_output_names)) {
    return false;
  }

  Ort::ConstNode softmax_node{nullptr};
  if (!GetProducer(unsqueeze_inputs[0], softmax_node) ||
      !IsOnnxOp(softmax_node, "Softmax") ||
      accepted_node_ids.count(softmax_node.GetId()) != 0) {
    return false;
  }
  auto softmax_inputs = softmax_node.GetInputs();
  auto softmax_outputs = softmax_node.GetOutputs();
  if (softmax_inputs.size() != 1 || softmax_outputs.size() != 1 ||
      Name(softmax_outputs[0]) != Name(unsqueeze_inputs[0]) ||
      !HasSingleConsumer(softmax_outputs[0], graph_output_names) ||
      !IsLastAxisSoftmax(softmax_node, softmax_inputs[0])) {
    return false;
  }

  Ort::ConstNode temperature_node{nullptr};
  if (!GetProducer(softmax_inputs[0], temperature_node) ||
      (!IsOnnxOp(temperature_node, "Mul") &&
       !IsOnnxOp(temperature_node, "Div")) ||
      accepted_node_ids.count(temperature_node.GetId()) != 0) {
    return false;
  }
  auto temperature_inputs = temperature_node.GetInputs();
  auto temperature_outputs = temperature_node.GetOutputs();
  if (temperature_inputs.size() != 2 || temperature_outputs.size() != 1 ||
      Name(temperature_outputs[0]) != Name(softmax_inputs[0]) ||
      !HasSingleConsumer(temperature_outputs[0], graph_output_names)) {
    return false;
  }

  Ort::ConstNode add_node{nullptr};
  int64_t temp_data_index = -1;
  const int64_t temperature_data_end =
      IsOnnxOp(temperature_node, "Div") ? 1 : 2;
  for (int64_t i = 0; i < temperature_data_end; ++i) {
    Ort::ConstNode producer{nullptr};
    if (GetProducer(temperature_inputs[static_cast<size_t>(i)], producer) &&
        IsOnnxOp(producer, "Add")) {
      add_node = producer;
      temp_data_index = i;
      break;
    }
  }
  if (!add_node || accepted_node_ids.count(add_node.GetId()) != 0) {
    return false;
  }
  size_t temp_scalar_index = 0;
  if (!HasConstantInitializerOtherInput(temperature_node,
                                        static_cast<size_t>(temp_data_index),
                                        &temp_scalar_index)) {
    return false;
  }
  auto add_inputs = add_node.GetInputs();
  auto add_outputs = add_node.GetOutputs();
  if (add_inputs.size() != 2 || add_outputs.size() != 1 ||
      Name(add_outputs[0]) !=
          Name(temperature_inputs[static_cast<size_t>(temp_data_index)]) ||
      !HasSingleConsumer(add_outputs[0], graph_output_names)) {
    return false;
  }

  Ort::ConstNode scale_mul_node{nullptr};
  Ort::ConstNode score_einsum{nullptr};
  int64_t add_score_index = -1;
  int64_t scale_data_index = -1;
  for (int64_t i = 0; i < 2; ++i) {
    Ort::ConstNode producer{nullptr};
    if (!GetProducer(add_inputs[static_cast<size_t>(i)], producer) ||
        !IsOnnxOp(producer, "Mul")) {
      continue;
    }
    auto candidate_inputs = producer.GetInputs();
    if (candidate_inputs.size() != 2) {
      continue;
    }
    for (int64_t j = 0; j < 2; ++j) {
      Ort::ConstNode candidate_score{nullptr};
      size_t scalar_index = 0;
      if (GetProducer(candidate_inputs[static_cast<size_t>(j)],
                      candidate_score) &&
          IsOnnxOp(candidate_score, "Einsum") &&
          HasConstantInitializerOtherInput(producer, static_cast<size_t>(j),
                                           &scalar_index)) {
        scale_mul_node = producer;
        score_einsum = candidate_score;
        add_score_index = i;
        scale_data_index = j;
        break;
      }
    }
    if (scale_mul_node) {
      break;
    }
  }
  if (!scale_mul_node || accepted_node_ids.count(scale_mul_node.GetId()) != 0) {
    return false;
  }
  auto scale_mul_inputs = scale_mul_node.GetInputs();
  auto scale_mul_outputs = scale_mul_node.GetOutputs();
  if (scale_mul_inputs.size() != 2 || scale_mul_outputs.size() != 1 ||
      Name(scale_mul_outputs[0]) !=
          Name(add_inputs[static_cast<size_t>(add_score_index)]) ||
      !HasSingleConsumer(scale_mul_outputs[0], graph_output_names)) {
    return false;
  }

  if (!score_einsum || scale_data_index < 0 ||
      accepted_node_ids.count(score_einsum.GetId()) != 0) {
    return false;
  }
  size_t scale_scalar_index = 0;
  if (!HasConstantInitializerOtherInput(scale_mul_node,
                                        static_cast<size_t>(scale_data_index),
                                        &scale_scalar_index)) {
    return false;
  }
  auto equation = GetStringAttribute(score_einsum, "equation");
  auto score_inputs = score_einsum.GetInputs();
  auto score_outputs = score_einsum.GetOutputs();
  if (!equation.has_value() ||
      !musa_ep::IsSupportedMhtaSimRank3Equation(*equation) ||
      score_inputs.size() != 2 || score_outputs.size() != 1 ||
      Name(score_outputs[0]) !=
          Name(scale_mul_inputs[static_cast<size_t>(scale_data_index)]) ||
      !HasSingleConsumer(score_outputs[0], graph_output_names)) {
    return false;
  }

  if (!ShapesAreSupportedForSimRank3(score_inputs[1], score_inputs[0],
                                     value_inputs[1], reshape_outputs[0])) {
    return false;
  }

  fusion_nodes = {score_einsum, scale_mul_node, add_node,     temperature_node,
                  softmax_node, unsqueeze_node, value_matmul, output_reshape};
  return true;
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>>
FindMhtaScaledDotProductAttentionFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (CanFuseBooleanMhtaScaledDotProductAttention(
            node, graph_output_names, accepted_node_ids, fusion_nodes) ||
        CanFuseLseqPreMaskMhtaScaledDotProductAttention(
            node, graph_output_names, accepted_node_ids, fusion_nodes) ||
        CanFuseMhtaScaledDotProductAttention(node, graph_output_names,
                                             accepted_node_ids, fusion_nodes) ||
        CanFuseSimRank3MhtaScaledDotProductAttention(
            node, graph_output_names, accepted_node_ids, fusion_nodes)) {
      fusions.push_back(std::move(fusion_nodes));
    }
  }
  return fusions;
}

}  // namespace musa_ep
