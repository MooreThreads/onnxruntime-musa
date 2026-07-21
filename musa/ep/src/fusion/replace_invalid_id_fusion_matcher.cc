// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

ONNXTensorElementDataType ElementType(Ort::ConstValueInfo value_info) {
  return value_info.TypeInfo().GetTensorTypeAndShapeInfo().GetElementType();
}

bool IsScalarIntegerInitializerOfType(Ort::ConstValueInfo value_info,
                                      ONNXTensorElementDataType element_type) {
  if (!IsSmallIntegerInitializer(value_info)) {
    return false;
  }
  auto info = value_info.TypeInfo().GetTensorTypeAndShapeInfo();
  return info.GetElementCount() == 1 && info.GetElementType() == element_type;
}

bool IsCastTo(Ort::ConstNode cast_node,
              ONNXTensorElementDataType element_type) {
  std::optional<int64_t> to = GetIntAttribute(cast_node, "to");
  return to.has_value() && *to == static_cast<int64_t>(element_type);
}

bool IsInternalValue(
    Ort::ConstValueInfo value_info,
    const std::unordered_set<std::string>& graph_output_names) {
  return graph_output_names.count(Name(value_info)) == 0;
}

bool HasOnlyConsumers(Ort::ConstValueInfo output,
                      std::array<Ort::ConstNode, 2> expected_nodes) {
  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
      output.GetConsumers();
  if (consumers.size() != expected_nodes.size()) {
    return false;
  }
  std::unordered_set<size_t> expected_ids = {expected_nodes[0].GetId(),
                                             expected_nodes[1].GetId()};
  for (const auto& consumer : consumers) {
    if (expected_ids.count(consumer.node.GetId()) == 0) {
      return false;
    }
  }
  return true;
}

struct Branch {
  Ort::ConstNode mul{nullptr};
  Ort::ConstNode cast{nullptr};
  Ort::ConstNode source{nullptr};
  Ort::ConstValueInfo other_input{nullptr};
};

bool ParseBranch(
    Ort::ConstNode mul_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstNode add_node, int64_t add_input_index,
    const std::unordered_set<std::string>& graph_output_names,
    ONNXTensorElementDataType element_type, Branch& branch) {
  if (!IsOnnxOp(mul_node, "Mul")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> inputs = mul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> outputs = mul_node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1 ||
      !IsInternalValue(outputs[0], graph_output_names) ||
      !HasOnlyConsumer(outputs[0], add_node, add_input_index) ||
      ElementType(outputs[0]) != element_type) {
    return false;
  }

  for (Ort::ConstValueInfo input : inputs) {
    Ort::ConstNode producer = FindProducer(producers, input);
    if (IsOnnxOp(producer, "Cast")) {
      if (branch.cast) {
        return false;
      }
      branch.cast = producer;
    } else {
      if (branch.other_input != nullptr) {
        return false;
      }
      branch.other_input = input;
    }
  }
  if (!branch.cast || branch.other_input == nullptr ||
      !IsCastTo(branch.cast, element_type)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> cast_inputs = branch.cast.GetInputs();
  std::vector<Ort::ConstValueInfo> cast_outputs = branch.cast.GetOutputs();
  if (cast_inputs.size() != 1 || cast_outputs.size() != 1 ||
      !IsInternalValue(cast_outputs[0], graph_output_names) ||
      !HasOnlyConsumer(cast_outputs[0], mul_node,
                       Name(inputs[0]) == Name(cast_outputs[0]) ? 0 : 1) ||
      ElementType(cast_outputs[0]) != element_type) {
    return false;
  }

  branch.mul = mul_node;
  branch.source = FindProducer(producers, cast_inputs[0]);
  return IsOnnxOp(branch.source, "LessOrEqual") ||
         IsOnnxOp(branch.source, "Not");
}

bool CanFuseReplaceInvalidId(
    Ort::ConstNode add_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(add_node, "Add") ||
      accepted_node_ids.count(add_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  if (add_inputs.size() != 2 || add_outputs.size() != 1 ||
      !IsIntTensorValueInfo(add_outputs[0])) {
    return false;
  }
  const ONNXTensorElementDataType element_type = ElementType(add_outputs[0]);

  Branch branches[2];
  for (int64_t i = 0; i < 2; ++i) {
    if (!ParseBranch(FindProducer(producers, add_inputs[i]), producers,
                     add_node, i, graph_output_names, element_type,
                     branches[i])) {
      return false;
    }
  }
  Branch* replacement_branch = nullptr;
  Branch* keep_branch = nullptr;
  for (Branch& branch : branches) {
    if (IsOnnxOp(branch.source, "LessOrEqual")) {
      if (replacement_branch != nullptr) {
        return false;
      }
      replacement_branch = &branch;
    } else {
      if (keep_branch != nullptr) {
        return false;
      }
      keep_branch = &branch;
    }
  }
  if (replacement_branch == nullptr || keep_branch == nullptr ||
      !IsScalarIntegerInitializerOfType(replacement_branch->other_input,
                                        element_type)) {
    return false;
  }
  Ort::ConstNode less_equal_node = replacement_branch->source;
  Ort::ConstNode not_node = keep_branch->source;
  std::vector<Ort::ConstValueInfo> not_inputs = not_node.GetInputs();
  std::vector<Ort::ConstValueInfo> not_outputs = not_node.GetOutputs();
  Ort::ConstNode not_input_producer =
      not_inputs.size() == 1 ? FindProducer(producers, not_inputs[0])
                             : Ort::ConstNode{nullptr};
  if (not_inputs.size() != 1 || not_outputs.size() != 1 ||
      !not_input_producer ||
      not_input_producer.GetId() != less_equal_node.GetId() ||
      !IsInternalValue(not_outputs[0], graph_output_names) ||
      !HasOnlyConsumer(not_outputs[0], keep_branch->cast, 0)) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> less_equal_inputs =
      less_equal_node.GetInputs();
  std::vector<Ort::ConstValueInfo> less_equal_outputs =
      less_equal_node.GetOutputs();
  if (less_equal_inputs.size() != 2 || less_equal_outputs.size() != 1 ||
      !IsInternalValue(less_equal_outputs[0], graph_output_names) ||
      !HasOnlyConsumers(less_equal_outputs[0],
                        {not_node, replacement_branch->cast}) ||
      !IsIntTensorValueInfo(less_equal_inputs[0]) ||
      ElementType(less_equal_inputs[0]) != element_type ||
      IsConstantInitializerValueInfo(less_equal_inputs[0]) ||
      !IsScalarIntegerInitializerOfType(less_equal_inputs[1], element_type) ||
      Name(less_equal_inputs[0]) != Name(keep_branch->other_input)) {
    return false;
  }
  auto input_shape = GetTensorShape(less_equal_inputs[0]);
  auto output_shape = GetTensorShape(add_outputs[0]);
  if (input_shape.has_value() && output_shape.has_value() &&
      !ShapesEqualOnKnownDims(*input_shape, *output_shape)) {
    return false;
  }
  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  for (Ort::ConstNode node :
       {less_equal_node, not_node, keep_branch->cast, keep_branch->mul,
        replacement_branch->cast, replacement_branch->mul, add_node}) {
    if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindReplaceInvalidIdFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);
  for (Ort::ConstNode add_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (CanFuseReplaceInvalidId(add_node, producers, graph_output_names,
                                accepted_node_ids, fusion_nodes)) {
      fusions.push_back(std::move(fusion_nodes));
    }
  }
  return fusions;
}

}  // namespace musa_ep
