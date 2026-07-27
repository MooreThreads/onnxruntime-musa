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
#include <unordered_set>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"

namespace musa_ep {

namespace {

bool CanFuseConcatReshape(
    Ort::ConstNode reshape_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(reshape_node, "Reshape") ||
      accepted_node_ids.count(reshape_node.GetId()) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reshape_outputs = reshape_node.GetOutputs();
  if (reshape_inputs.size() != 2 || reshape_outputs.size() != 1 ||
      !IsSmallIntegerInitializer(reshape_inputs[1])) {
    return false;
  }

  Ort::ConstNode unsqueeze_node{nullptr};
  Ort::ConstValueInfo concat_or_unsqueeze_output = reshape_inputs[0];
  Ort::ConstNode concat_node = FindProducer(producers, reshape_inputs[0]);
  if (IsOnnxOp(concat_node, "Unsqueeze")) {
    unsqueeze_node = concat_node;
    if (accepted_node_ids.count(unsqueeze_node.GetId()) != 0) {
      return false;
    }
    std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
        unsqueeze_node.GetInputs();
    std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
        unsqueeze_node.GetOutputs();
    if (unsqueeze_inputs.size() != 2 || unsqueeze_outputs.size() != 1 ||
        graph_output_names.count(Name(unsqueeze_outputs[0])) != 0 ||
        !IsSmallIntegerInitializer(unsqueeze_inputs[1]) ||
        !HasOnlyConsumer(unsqueeze_outputs[0], reshape_node, 0)) {
      return false;
    }
    concat_node = FindProducer(producers, unsqueeze_inputs[0]);
    concat_or_unsqueeze_output = unsqueeze_inputs[0];
  }

  if (!IsOnnxOp(concat_node, "Concat") ||
      accepted_node_ids.count(concat_node.GetId()) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_inputs.empty() || concat_outputs.size() != 1 ||
      graph_output_names.count(Name(concat_outputs[0])) != 0 ||
      !HasOnlyConsumer(concat_outputs[0],
                       unsqueeze_node ? unsqueeze_node : reshape_node, 0) ||
      Name(concat_outputs[0]) != Name(concat_or_unsqueeze_output)) {
    return false;
  }
  for (Ort::ConstValueInfo concat_input : concat_inputs) {
    if (concat_input == nullptr || concat_input.IsConstantInitializer() ||
        IsOnnxOp(FindProducer(producers, concat_input), "Constant")) {
      return false;
    }
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  if (!AddFusionNode(concat_node, accepted_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }
  if (unsqueeze_node && !AddFusionNode(unsqueeze_node, accepted_node_ids,
                                       selected_node_ids, fusion_nodes)) {
    return false;
  }
  if (!AddFusionNode(reshape_node, accepted_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                     selected_node_ids);
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindConcatReshapeFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);

  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseConcatReshape(node, producers, graph_output_names,
                              accepted_node_ids, fusion_nodes)) {
      continue;
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

}  // namespace musa_ep
