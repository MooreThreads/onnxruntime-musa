// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
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

bool CanFuseCenteredReduce(
    Ort::ConstNode first_reduce_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsReduceSumOrProd(first_reduce_node) ||
      fused_node_ids.count(first_reduce_node.GetId()) != 0 ||
      GetIntAttribute(first_reduce_node, "keepdims").value_or(1) != 1) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> first_reduce_inputs =
      first_reduce_node.GetInputs();
  std::vector<Ort::ConstValueInfo> first_reduce_outputs =
      first_reduce_node.GetOutputs();
  if (first_reduce_inputs.size() < 2 || first_reduce_outputs.size() != 1 ||
      !IsFloatTensorValueInfo(first_reduce_inputs[0]) ||
      !IsFloatTensorValueInfo(first_reduce_outputs[0])) {
    return false;
  }

  auto input_shape = GetTensorShape(first_reduce_inputs[0]);
  if (!input_shape.has_value() || input_shape->size() < 2 ||
      input_shape->back() <= 0 ||
      !ReduceAxesAreLastDim(first_reduce_node, input_shape->size()) ||
      !ReduceOutputKeepsLastDim(first_reduce_outputs[0], *input_shape)) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> first_consumers =
      first_reduce_outputs[0].GetConsumers();
  Ort::ConstNode sub_node{nullptr};
  for (const auto& consumer : first_consumers) {
    if (consumer.index == 1 && IsOnnxOp(consumer.node, "Sub")) {
      sub_node = consumer.node;
      break;
    }
  }
  if (!sub_node || fused_node_ids.count(sub_node.GetId()) != 0 ||
      !ValueHasExternalConsumerOrGraphOutput(first_reduce_outputs[0], sub_node,
                                             graph_output_names)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> sub_inputs = sub_node.GetInputs();
  std::vector<Ort::ConstValueInfo> sub_outputs = sub_node.GetOutputs();
  if (sub_inputs.size() != 2 || sub_outputs.size() != 1 ||
      graph_output_names.count(Name(sub_outputs[0])) != 0 ||
      Name(sub_inputs[0]) != Name(first_reduce_inputs[0]) ||
      Name(sub_inputs[1]) != Name(first_reduce_outputs[0]) ||
      !IsFloatTensorValueInfo(sub_outputs[0])) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> sub_consumers =
      sub_outputs[0].GetConsumers();
  Ort::ConstNode mul_node{nullptr};
  for (const auto& consumer : sub_consumers) {
    if (IsOnnxOp(consumer.node, "Mul")) {
      mul_node = consumer.node;
      break;
    }
  }
  if (!mul_node || fused_node_ids.count(mul_node.GetId()) != 0 ||
      !ValueHasOnlyConsumers(sub_outputs[0], mul_node)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> mul_inputs = mul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> mul_outputs = mul_node.GetOutputs();
  if (mul_inputs.size() != 2 || mul_outputs.size() != 1 ||
      graph_output_names.count(Name(mul_outputs[0])) != 0 ||
      Name(mul_inputs[0]) != Name(sub_outputs[0]) ||
      Name(mul_inputs[1]) != Name(sub_outputs[0]) ||
      !IsFloatTensorValueInfo(mul_outputs[0])) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> mul_consumers =
      mul_outputs[0].GetConsumers();
  Ort::ConstNode second_reduce_node{nullptr};
  for (const auto& consumer : mul_consumers) {
    if (consumer.index == 0 && IsReduceSumOrProd(consumer.node)) {
      second_reduce_node = consumer.node;
      break;
    }
  }
  if (!second_reduce_node ||
      fused_node_ids.count(second_reduce_node.GetId()) != 0 ||
      !ValueHasOnlyConsumers(mul_outputs[0], second_reduce_node) ||
      GetIntAttribute(second_reduce_node, "keepdims").value_or(1) != 1) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> second_reduce_inputs =
      second_reduce_node.GetInputs();
  std::vector<Ort::ConstValueInfo> second_reduce_outputs =
      second_reduce_node.GetOutputs();
  if (second_reduce_inputs.size() < 2 || second_reduce_outputs.size() != 1 ||
      !ReduceAxesAreLastDim(second_reduce_node, input_shape->size()) ||
      !IsFloatTensorValueInfo(second_reduce_outputs[0]) ||
      !ReduceOutputKeepsLastDim(second_reduce_outputs[0], *input_shape)) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  for (Ort::ConstNode node :
       {first_reduce_node, sub_node, mul_node, second_reduce_node}) {
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

std::vector<std::vector<Ort::ConstNode>> FindCenteredReduceFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode first_reduce_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseCenteredReduce(first_reduce_node, graph_output_names,
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
