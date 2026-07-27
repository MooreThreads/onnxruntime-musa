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

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {

bool CanFuseRmsNorm(
    Ort::ConstNode output_mul_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(output_mul_node, "Mul") ||
      accepted_node_ids.count(output_mul_node.GetId()) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> output_mul_inputs =
      output_mul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> output_mul_outputs =
      output_mul_node.GetOutputs();
  if (output_mul_inputs.size() != 2 || output_mul_outputs.size() != 1 ||
      !IsFloatTensorValueInfo(output_mul_outputs[0])) {
    return false;
  }

  Ort::ConstNode div_node{nullptr};
  Ort::ConstValueInfo gamma_input{nullptr};
  auto producer_it = producers.find(Name(output_mul_inputs[0]));
  if (producer_it != producers.end() && IsOnnxOp(producer_it->second, "Div")) {
    div_node = producer_it->second;
    gamma_input = output_mul_inputs[1];
  } else {
    producer_it = producers.find(Name(output_mul_inputs[1]));
    if (producer_it != producers.end() &&
        IsOnnxOp(producer_it->second, "Div")) {
      div_node = producer_it->second;
      gamma_input = output_mul_inputs[0];
    }
  }
  if (!div_node || accepted_node_ids.count(div_node.GetId()) != 0 ||
      !IsFloatTensorValueInfo(gamma_input)) {
    return false;
  }

  auto gamma_shape = GetTensorShape(gamma_input);
  if (!gamma_shape.has_value() || gamma_shape->size() != 1 ||
      (*gamma_shape)[0] <= 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> div_inputs = div_node.GetInputs();
  std::vector<Ort::ConstValueInfo> div_outputs = div_node.GetOutputs();
  if (div_inputs.size() != 2 || div_outputs.size() != 1 ||
      graph_output_names.count(Name(div_outputs[0])) != 0 ||
      !IsFloatTensorValueInfo(div_inputs[0]) ||
      !IsFloatTensorValueInfo(div_outputs[0]) ||
      !ValueHasOnlyConsumers(div_outputs[0], output_mul_node)) {
    return false;
  }

  auto input_shape = GetTensorShape(div_inputs[0]);
  if (!input_shape.has_value() || input_shape->size() < 2 ||
      (input_shape->back() > 0 && input_shape->back() != (*gamma_shape)[0])) {
    return false;
  }
  auto output_shape = GetTensorShape(output_mul_outputs[0]);
  if (output_shape.has_value() &&
      (output_shape->size() != input_shape->size() ||
       output_shape->back() != (*gamma_shape)[0])) {
    return false;
  }

  producer_it = producers.find(Name(div_inputs[1]));
  if (producer_it == producers.end() ||
      !IsOnnxOp(producer_it->second, "Sqrt")) {
    return false;
  }
  Ort::ConstNode sqrt_node = producer_it->second;
  std::vector<Ort::ConstValueInfo> sqrt_inputs = sqrt_node.GetInputs();
  std::vector<Ort::ConstValueInfo> sqrt_outputs = sqrt_node.GetOutputs();
  if (accepted_node_ids.count(sqrt_node.GetId()) != 0 ||
      sqrt_inputs.size() != 1 || sqrt_outputs.size() != 1 ||
      graph_output_names.count(Name(sqrt_outputs[0])) != 0 ||
      !IsFloatTensorValueInfo(sqrt_outputs[0]) ||
      !HasOnlyConsumer(sqrt_outputs[0], div_node, 1)) {
    return false;
  }

  producer_it = producers.find(Name(sqrt_inputs[0]));
  if (producer_it == producers.end() || !IsOnnxOp(producer_it->second, "Add")) {
    return false;
  }
  Ort::ConstNode add_node = producer_it->second;
  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  if (accepted_node_ids.count(add_node.GetId()) != 0 ||
      add_inputs.size() != 2 || add_outputs.size() != 1 ||
      graph_output_names.count(Name(add_outputs[0])) != 0 ||
      !IsFloatTensorValueInfo(add_outputs[0]) ||
      !HasOnlyConsumer(add_outputs[0], sqrt_node, 0)) {
    return false;
  }

  Ort::ConstNode reduce_node{nullptr};
  Ort::ConstValueInfo epsilon_input{nullptr};
  producer_it = producers.find(Name(add_inputs[0]));
  if (producer_it != producers.end() &&
      IsOnnxOp(producer_it->second, "ReduceMean")) {
    reduce_node = producer_it->second;
    epsilon_input = add_inputs[1];
  } else {
    producer_it = producers.find(Name(add_inputs[1]));
    if (producer_it != producers.end() &&
        IsOnnxOp(producer_it->second, "ReduceMean")) {
      reduce_node = producer_it->second;
      epsilon_input = add_inputs[0];
    }
  }
  if (!reduce_node || accepted_node_ids.count(reduce_node.GetId()) != 0 ||
      !ReadScalarFloatInitializer(epsilon_input).has_value()) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> reduce_inputs = reduce_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reduce_outputs = reduce_node.GetOutputs();
  if (reduce_inputs.size() < 2 || reduce_outputs.size() != 1 ||
      GetIntAttribute(reduce_node, "keepdims").value_or(1) != 1 ||
      graph_output_names.count(Name(reduce_outputs[0])) != 0 ||
      !IsFloatTensorValueInfo(reduce_outputs[0]) ||
      !ReduceAxesAreLastDim(reduce_node, input_shape->size()) ||
      !ReduceOutputKeepsLastDim(reduce_outputs[0], *input_shape) ||
      !ValueHasOnlyConsumers(reduce_outputs[0], add_node)) {
    return false;
  }

  producer_it = producers.find(Name(reduce_inputs[0]));
  if (producer_it == producers.end() || !IsOnnxOp(producer_it->second, "Mul")) {
    return false;
  }
  Ort::ConstNode square_node = producer_it->second;
  std::vector<Ort::ConstValueInfo> square_inputs = square_node.GetInputs();
  std::vector<Ort::ConstValueInfo> square_outputs = square_node.GetOutputs();
  if (accepted_node_ids.count(square_node.GetId()) != 0 ||
      square_inputs.size() != 2 || square_outputs.size() != 1 ||
      graph_output_names.count(Name(square_outputs[0])) != 0 ||
      Name(square_inputs[0]) != Name(div_inputs[0]) ||
      Name(square_inputs[1]) != Name(div_inputs[0]) ||
      !IsFloatTensorValueInfo(square_outputs[0]) ||
      !HasOnlyConsumer(square_outputs[0], reduce_node, 0)) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  for (Ort::ConstNode node : {square_node, reduce_node, add_node, sqrt_node,
                              div_node, output_mul_node}) {
    if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }
  if (!FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                   selected_node_ids)) {
    return false;
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindRmsNormFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);
  for (Ort::ConstNode output_mul_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseRmsNorm(output_mul_node, producers, graph_output_names,
                        accepted_node_ids, fusion_nodes)) {
      continue;
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

}  // namespace musa_ep
