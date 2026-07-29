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
      !IsFloatTensorValueInfo(value_info)) {
    return false;
  }
  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK()) {
    return false;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  const float* data = value.GetTensorData<float>();
  const size_t count = info.GetElementCount();
  for (size_t i = 0; i < count; ++i) {
    if (std::fabs(data[i]) > 0.0f) {
      return false;
    }
  }
  return true;
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
  if (!IsFloatTensorValueInfo(q) || !IsFloatTensorValueInfo(k) ||
      !IsFloatTensorValueInfo(v) || !IsFloatTensorValueInfo(output)) {
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

bool ShapesAreSupportedForSimRank3(Ort::ConstValueInfo q, Ort::ConstValueInfo k,
                                   Ort::ConstValueInfo v,
                                   Ort::ConstValueInfo output) {
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
    if (CanFuseMhtaScaledDotProductAttention(node, graph_output_names,
                                             accepted_node_ids, fusion_nodes) ||
        CanFuseSimRank3MhtaScaledDotProductAttention(
            node, graph_output_names, accepted_node_ids, fusion_nodes)) {
      fusions.push_back(std::move(fusion_nodes));
    }
  }
  return fusions;
}

}  // namespace musa_ep
