// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

bool CanFuseMathConcatLog(
    Ort::ConstNode output_mul_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(output_mul_node, "Mul") ||
      fused_node_ids.count(output_mul_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> mul_inputs = output_mul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> mul_outputs = output_mul_node.GetOutputs();
  if (mul_inputs.size() != 2 || mul_outputs.size() != 1 ||
      !IsFloatTensorValueInfo(mul_outputs[0])) {
    return false;
  }

  Ort::ConstNode log_node{nullptr};
  Ort::ConstValueInfo scale_input{nullptr};
  for (Ort::ConstValueInfo input : mul_inputs) {
    if (IsConstantInitializerValueInfo(input)) {
      scale_input = input;
    } else {
      Ort::ConstNode producer = FindProducer(producers, input);
      if (IsOnnxOp(producer, "Log")) {
        log_node = producer;
      } else {
        scale_input = input;
      }
    }
  }
  if (!log_node || fused_node_ids.count(log_node.GetId()) != 0 ||
      !IsConstantInitializerValueInfo(scale_input)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> log_inputs = log_node.GetInputs();
  std::vector<Ort::ConstValueInfo> log_outputs = log_node.GetOutputs();
  if (log_inputs.size() != 1 || log_outputs.size() != 1 ||
      graph_output_names.count(Name(log_outputs[0])) != 0 ||
      !IsFloatTensorValueInfo(log_outputs[0]) ||
      !ValueHasOnlyConsumers(log_outputs[0], output_mul_node)) {
    return false;
  }

  Ort::ConstNode add_node = FindProducer(producers, log_inputs[0]);
  if (!IsOnnxOp(add_node, "Add") ||
      fused_node_ids.count(add_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  if (add_inputs.size() != 2 || add_outputs.size() != 1 ||
      graph_output_names.count(Name(add_outputs[0])) != 0 ||
      !IsFloatTensorValueInfo(add_outputs[0]) ||
      !HasOnlyConsumer(add_outputs[0], log_node, 0)) {
    return false;
  }

  Ort::ConstNode max_node{nullptr};
  Ort::ConstValueInfo add_input{nullptr};
  for (Ort::ConstValueInfo input : add_inputs) {
    if (IsConstantInitializerValueInfo(input)) {
      add_input = input;
    } else {
      Ort::ConstNode producer = FindProducer(producers, input);
      if (IsOnnxOp(producer, "Max")) {
        max_node = producer;
      } else {
        add_input = input;
      }
    }
  }
  if (!max_node || fused_node_ids.count(max_node.GetId()) != 0 ||
      !IsConstantInitializerValueInfo(add_input)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> max_inputs = max_node.GetInputs();
  std::vector<Ort::ConstValueInfo> max_outputs = max_node.GetOutputs();
  if (max_inputs.size() != 2 || max_outputs.size() != 1 ||
      graph_output_names.count(Name(max_outputs[0])) != 0 ||
      !IsFloatTensorValueInfo(max_outputs[0]) ||
      !HasOnlyConsumer(max_outputs[0], add_node,
                       Name(add_inputs[0]) == Name(max_outputs[0]) ? 0 : 1)) {
    return false;
  }

  Ort::ConstValueInfo input{nullptr};
  Ort::ConstValueInfo max_input{nullptr};
  for (Ort::ConstValueInfo candidate : max_inputs) {
    if (IsConstantInitializerValueInfo(candidate)) {
      max_input = candidate;
    } else {
      input = candidate;
    }
  }
  if (input == nullptr || max_input == nullptr ||
      !IsFloatTensorValueInfo(input) ||
      !IsConstantInitializerValueInfo(max_input)) {
    return false;
  }

  auto input_shape = GetTensorShape(input);
  auto output_shape = GetTensorShape(mul_outputs[0]);
  if (input_shape.has_value() && output_shape.has_value() &&
      !ShapesEqualOnKnownDims(*input_shape, *output_shape)) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  for (Ort::ConstNode node : {max_node, add_node, log_node, output_mul_node}) {
    if (!AddFusionNode(node, fused_node_ids, selected_node_ids, fusion_nodes)) {
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

std::vector<std::vector<Ort::ConstNode>> FindMathConcatLogFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);
  for (Ort::ConstNode output_mul_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseMathConcatLog(output_mul_node, producers, graph_output_names,
                              fused_node_ids, fusion_nodes)) {
      continue;
    }
    for (Ort::ConstNode node : fusion_nodes) {
      fused_node_ids.insert(node.GetId());
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

}  // namespace musa_ep
