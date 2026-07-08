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

bool CanFuseModuloGather(
    Ort::ConstNode gather_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(gather_node, "Gather") ||
      accepted_node_ids.count(gather_node.GetId()) != 0 ||
      GetIntAttribute(gather_node, "axis").value_or(0) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gather_outputs = gather_node.GetOutputs();
  if (gather_inputs.size() != 2 || gather_outputs.size() != 1 ||
      !IsFloatTensorValueInfo(gather_outputs[0])) {
    return false;
  }

  auto final_mul_it = producers.find(Name(gather_inputs[1]));
  if (final_mul_it == producers.end() ||
      !IsOnnxOp(final_mul_it->second, "Mul")) {
    return false;
  }
  Ort::ConstNode final_mul = final_mul_it->second;
  if (accepted_node_ids.count(final_mul.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> final_mul_inputs = final_mul.GetInputs();
  std::vector<Ort::ConstValueInfo> final_mul_outputs = final_mul.GetOutputs();
  if (final_mul_inputs.size() != 2 || final_mul_outputs.size() != 1 ||
      graph_output_names.count(Name(final_mul_outputs[0])) != 0 ||
      !HasOnlyConsumer(final_mul_outputs[0], gather_node, 1) ||
      !IsIntTensorValueInfo(final_mul_outputs[0])) {
    return false;
  }

  Ort::ConstNode add_node{nullptr};
  Ort::ConstNode cast_node{nullptr};
  for (Ort::ConstValueInfo input : final_mul_inputs) {
    auto it = producers.find(Name(input));
    if (it != producers.end() && IsOnnxOp(it->second, "Add")) {
      add_node = it->second;
    } else if (it != producers.end() && IsOnnxOp(it->second, "Cast")) {
      cast_node = it->second;
    }
  }
  if (!add_node || !cast_node ||
      accepted_node_ids.count(add_node.GetId()) != 0 ||
      accepted_node_ids.count(cast_node.GetId()) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  if (add_inputs.size() != 2 || add_outputs.size() != 1 ||
      graph_output_names.count(Name(add_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(add_outputs[0], final_mul) ||
      !IsIntTensorValueInfo(add_outputs[0])) {
    return false;
  }
  Ort::ConstNode sub_node{nullptr};
  Ort::ConstValueInfo offset_input{nullptr};
  for (Ort::ConstValueInfo input : add_inputs) {
    auto it = producers.find(Name(input));
    if (it != producers.end() && IsOnnxOp(it->second, "Sub")) {
      sub_node = it->second;
    } else {
      offset_input = input;
    }
  }
  std::optional<int64_t> offset = ReadScalarIntInitializer(offset_input);
  if (!sub_node || accepted_node_ids.count(sub_node.GetId()) != 0 ||
      !offset.has_value() || *offset < 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> sub_inputs = sub_node.GetInputs();
  std::vector<Ort::ConstValueInfo> sub_outputs = sub_node.GetOutputs();
  if (sub_inputs.size() != 2 || sub_outputs.size() != 1 ||
      graph_output_names.count(Name(sub_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(sub_outputs[0], add_node) ||
      !IsIntTensorValueInfo(sub_outputs[0])) {
    return false;
  }
  Ort::ConstValueInfo source_input = sub_inputs[0];
  auto product_it = producers.find(Name(sub_inputs[1]));
  if (product_it == producers.end() || !IsOnnxOp(product_it->second, "Mul")) {
    return false;
  }
  Ort::ConstNode product_mul = product_it->second;
  if (accepted_node_ids.count(product_mul.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> product_inputs = product_mul.GetInputs();
  std::vector<Ort::ConstValueInfo> product_outputs = product_mul.GetOutputs();
  if (product_inputs.size() != 2 || product_outputs.size() != 1 ||
      graph_output_names.count(Name(product_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(product_outputs[0], sub_node) ||
      !IsIntTensorValueInfo(product_outputs[0])) {
    return false;
  }

  Ort::ConstNode div_node{nullptr};
  Ort::ConstValueInfo modulus_input{nullptr};
  for (Ort::ConstValueInfo input : product_inputs) {
    auto it = producers.find(Name(input));
    if (it != producers.end() && IsOnnxOp(it->second, "Div")) {
      div_node = it->second;
    } else {
      modulus_input = input;
    }
  }
  std::optional<int64_t> modulus = ReadScalarIntInitializer(modulus_input);
  if (!div_node || accepted_node_ids.count(div_node.GetId()) != 0 ||
      !modulus.has_value() || *modulus <= 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> div_inputs = div_node.GetInputs();
  std::vector<Ort::ConstValueInfo> div_outputs = div_node.GetOutputs();
  if (div_inputs.size() != 2 || div_outputs.size() != 1 ||
      graph_output_names.count(Name(div_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(div_outputs[0], product_mul) ||
      Name(div_inputs[0]) != Name(source_input) ||
      Name(div_inputs[1]) != Name(modulus_input) ||
      !IsIntTensorValueInfo(div_outputs[0])) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> cast_inputs = cast_node.GetInputs();
  std::vector<Ort::ConstValueInfo> cast_outputs = cast_node.GetOutputs();
  if (cast_inputs.size() != 1 || cast_outputs.size() != 1 ||
      graph_output_names.count(Name(cast_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(cast_outputs[0], final_mul) ||
      !IsIntTensorValueInfo(cast_outputs[0])) {
    return false;
  }
  auto not_it = producers.find(Name(cast_inputs[0]));
  if (not_it == producers.end() || !IsOnnxOp(not_it->second, "Not")) {
    return false;
  }
  Ort::ConstNode not_node = not_it->second;
  if (accepted_node_ids.count(not_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> not_inputs = not_node.GetInputs();
  std::vector<Ort::ConstValueInfo> not_outputs = not_node.GetOutputs();
  if (not_inputs.size() != 1 || not_outputs.size() != 1 ||
      graph_output_names.count(Name(not_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(not_outputs[0], cast_node)) {
    return false;
  }
  auto equal_it = producers.find(Name(not_inputs[0]));
  if (equal_it == producers.end() || !IsOnnxOp(equal_it->second, "Equal")) {
    return false;
  }
  Ort::ConstNode equal_node = equal_it->second;
  if (accepted_node_ids.count(equal_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> equal_inputs = equal_node.GetInputs();
  std::vector<Ort::ConstValueInfo> equal_outputs = equal_node.GetOutputs();
  if (equal_inputs.size() != 2 || equal_outputs.size() != 1 ||
      graph_output_names.count(Name(equal_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(equal_outputs[0], not_node)) {
    return false;
  }
  Ort::ConstValueInfo invalid_input{nullptr};
  if (Name(equal_inputs[0]) == Name(source_input)) {
    invalid_input = equal_inputs[1];
  } else if (Name(equal_inputs[1]) == Name(source_input)) {
    invalid_input = equal_inputs[0];
  } else {
    return false;
  }
  if (!ReadScalarIntInitializer(invalid_input).has_value()) {
    return false;
  }

  std::optional<std::vector<int64_t>> table_shape =
      GetTensorShape(gather_inputs[0]);
  if (!table_shape.has_value() || table_shape->empty() ||
      (*table_shape)[0] <= 0 || *offset + *modulus > (*table_shape)[0]) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  for (Ort::ConstNode node :
       {equal_node, not_node, cast_node, div_node, product_mul, sub_node,
        add_node, final_mul, gather_node}) {
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

std::vector<std::vector<Ort::ConstNode>> FindModuloGatherFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);
  for (Ort::ConstNode gather_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseModuloGather(gather_node, producers, graph_output_names,
                             accepted_node_ids, fusion_nodes)) {
      continue;
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

}  // namespace musa_ep
