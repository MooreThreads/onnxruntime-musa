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

bool HasEinsumEquation(Ort::ConstNode node, const char* expected_equation) {
  if (!IsOnnxOp(node, "Einsum")) {
    return false;
  }
  auto equation = GetStringAttribute(node, "equation");
  return equation.has_value() && *equation == expected_equation;
}

struct ParallelEinsumActivationBranchMatch {
  Ort::ConstNode first_einsum{nullptr};
  Ort::ConstNode first_tanh{nullptr};
  Ort::ConstNode second_einsum{nullptr};
  Ort::ConstNode second_tanh{nullptr};
  Ort::ConstNode third_einsum{nullptr};
  Ort::ConstNode add{nullptr};
  Ort::ConstNode mul{nullptr};
  std::string mlp_input_name;
  std::string gate_input_name;
  std::string bias_name;
  std::vector<int64_t> w1_shape;
  std::vector<int64_t> w2_shape;
  std::vector<int64_t> w3_shape;
};

bool MatchParallelEinsumActivationBranch(
    Ort::ConstNode concat_node, size_t concat_input_index,
    Ort::ConstValueInfo concat_input,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    ParallelEinsumActivationBranchMatch& branch) {
  if (!HasSingleConsumerAt(concat_input, concat_node,
                           static_cast<int64_t>(concat_input_index),
                           graph_output_names) ||
      !IsFloatTensorValueInfo(concat_input)) {
    return false;
  }

  Ort::ConstNode mul_node{nullptr};
  if (!GetProducer(concat_input, mul_node) || !IsOnnxOp(mul_node, "Mul") ||
      fused_node_ids.count(mul_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> mul_inputs = mul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> mul_outputs = mul_node.GetOutputs();
  if (mul_inputs.size() != 2 || mul_outputs.size() != 1 ||
      Name(mul_outputs[0]) != Name(concat_input)) {
    return false;
  }

  Ort::ConstNode add_node{nullptr};
  int64_t add_input_index = -1;
  for (int64_t i = 0; i < 2; ++i) {
    Ort::ConstNode producer{nullptr};
    if (GetProducer(mul_inputs[static_cast<size_t>(i)], producer) &&
        IsOnnxOp(producer, "Add")) {
      add_node = producer;
      add_input_index = i;
      break;
    }
  }
  if (!add_node || fused_node_ids.count(add_node.GetId()) != 0) {
    return false;
  }
  const size_t gate_input_index = static_cast<size_t>(1 - add_input_index);
  if (!IsFloatTensorValueInfo(mul_inputs[gate_input_index])) {
    return false;
  }
  branch.gate_input_name = Name(mul_inputs[gate_input_index]);

  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  if (add_inputs.size() != 2 || add_outputs.size() != 1 ||
      Name(add_outputs[0]) !=
          Name(mul_inputs[static_cast<size_t>(add_input_index)]) ||
      !HasSingleConsumerAt(add_outputs[0], mul_node, add_input_index,
                           graph_output_names)) {
    return false;
  }

  Ort::ConstNode third_einsum{nullptr};
  int64_t third_input_index = -1;
  for (int64_t i = 0; i < 2; ++i) {
    Ort::ConstNode producer{nullptr};
    if (GetProducer(add_inputs[static_cast<size_t>(i)], producer) &&
        HasEinsumEquation(producer, "ij,bjk->bik")) {
      third_einsum = producer;
      third_input_index = i;
      break;
    }
  }
  if (!third_einsum || fused_node_ids.count(third_einsum.GetId()) != 0) {
    return false;
  }
  const size_t bias_input_index = static_cast<size_t>(1 - third_input_index);
  if (!IsFloatTensorValueInfo(add_inputs[bias_input_index])) {
    return false;
  }
  branch.bias_name = Name(add_inputs[bias_input_index]);

  std::vector<Ort::ConstValueInfo> third_inputs = third_einsum.GetInputs();
  std::vector<Ort::ConstValueInfo> third_outputs = third_einsum.GetOutputs();
  if (third_inputs.size() != 2 || third_outputs.size() != 1 ||
      Name(third_outputs[0]) !=
          Name(add_inputs[static_cast<size_t>(third_input_index)]) ||
      !HasSingleConsumerAt(third_outputs[0], add_node, third_input_index,
                           graph_output_names) ||
      !IsFloatTensorValueInfo(third_inputs[0]) ||
      !IsFloatTensorValueInfo(third_inputs[1])) {
    return false;
  }

  Ort::ConstNode second_tanh{nullptr};
  if (!GetProducer(third_inputs[1], second_tanh) ||
      !IsOnnxOp(second_tanh, "Tanh") ||
      fused_node_ids.count(second_tanh.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> second_tanh_inputs = second_tanh.GetInputs();
  std::vector<Ort::ConstValueInfo> second_tanh_outputs =
      second_tanh.GetOutputs();
  if (second_tanh_inputs.size() != 1 || second_tanh_outputs.size() != 1 ||
      Name(second_tanh_outputs[0]) != Name(third_inputs[1]) ||
      !HasSingleConsumerAt(second_tanh_outputs[0], third_einsum, 1,
                           graph_output_names)) {
    return false;
  }

  Ort::ConstNode second_einsum{nullptr};
  if (!GetProducer(second_tanh_inputs[0], second_einsum) ||
      !HasEinsumEquation(second_einsum, "ij,bjk->bik") ||
      fused_node_ids.count(second_einsum.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> second_inputs = second_einsum.GetInputs();
  std::vector<Ort::ConstValueInfo> second_outputs = second_einsum.GetOutputs();
  if (second_inputs.size() != 2 || second_outputs.size() != 1 ||
      Name(second_outputs[0]) != Name(second_tanh_inputs[0]) ||
      !HasSingleConsumerAt(second_outputs[0], second_tanh, 0,
                           graph_output_names) ||
      !IsFloatTensorValueInfo(second_inputs[0]) ||
      !IsFloatTensorValueInfo(second_inputs[1])) {
    return false;
  }

  Ort::ConstNode first_tanh{nullptr};
  if (!GetProducer(second_inputs[1], first_tanh) ||
      !IsOnnxOp(first_tanh, "Tanh") ||
      fused_node_ids.count(first_tanh.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> first_tanh_inputs = first_tanh.GetInputs();
  std::vector<Ort::ConstValueInfo> first_tanh_outputs = first_tanh.GetOutputs();
  if (first_tanh_inputs.size() != 1 || first_tanh_outputs.size() != 1 ||
      Name(first_tanh_outputs[0]) != Name(second_inputs[1]) ||
      !HasSingleConsumerAt(first_tanh_outputs[0], second_einsum, 1,
                           graph_output_names)) {
    return false;
  }

  Ort::ConstNode first_einsum{nullptr};
  if (!GetProducer(first_tanh_inputs[0], first_einsum) ||
      !HasEinsumEquation(first_einsum, "ij,bjk->bik") ||
      fused_node_ids.count(first_einsum.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> first_inputs = first_einsum.GetInputs();
  std::vector<Ort::ConstValueInfo> first_outputs = first_einsum.GetOutputs();
  if (first_inputs.size() != 2 || first_outputs.size() != 1 ||
      Name(first_outputs[0]) != Name(first_tanh_inputs[0]) ||
      !HasSingleConsumerAt(first_outputs[0], first_tanh, 0,
                           graph_output_names) ||
      !IsFloatTensorValueInfo(first_inputs[0]) ||
      !IsFloatTensorValueInfo(first_inputs[1])) {
    return false;
  }

  auto w1_shape = GetStaticShape(first_inputs[0]);
  auto w2_shape = GetStaticShape(second_inputs[0]);
  auto w3_shape = GetStaticShape(third_inputs[0]);
  auto input_shape = GetTensorShape(first_inputs[1]);
  auto gate_shape = GetTensorShape(mul_inputs[gate_input_index]);
  auto bias_shape = GetTensorShape(add_inputs[bias_input_index]);
  auto output_shape = GetTensorShape(mul_outputs[0]);
  if (!w1_shape.has_value() || !w2_shape.has_value() || !w3_shape.has_value() ||
      !input_shape.has_value() || !gate_shape.has_value() ||
      !bias_shape.has_value() || !output_shape.has_value() ||
      w1_shape->size() != 2 || w2_shape->size() != 2 || w3_shape->size() != 2 ||
      input_shape->size() != 3 || gate_shape->size() != 3 ||
      bias_shape->size() != 2 || output_shape->size() != 3 ||
      (*input_shape)[2] != 1 || (*gate_shape)[2] != 1 ||
      (*bias_shape)[1] != 1) {
    return false;
  }
  const int64_t hidden_dim = (*w1_shape)[0];
  const int64_t input_dim = (*w1_shape)[1];
  if (hidden_dim <= 0 || input_dim <= 0 || (*w2_shape)[0] != hidden_dim ||
      (*w2_shape)[1] != hidden_dim || (*w3_shape)[0] != input_dim ||
      (*w3_shape)[1] != hidden_dim || (*input_shape)[1] != input_dim ||
      (*gate_shape)[1] != input_dim || (*bias_shape)[0] != input_dim ||
      !ShapesEqualOnKnownDims(*gate_shape, *output_shape)) {
    return false;
  }

  branch.first_einsum = first_einsum;
  branch.first_tanh = first_tanh;
  branch.second_einsum = second_einsum;
  branch.second_tanh = second_tanh;
  branch.third_einsum = third_einsum;
  branch.add = add_node;
  branch.mul = mul_node;
  branch.mlp_input_name = Name(first_inputs[1]);
  branch.w1_shape = *w1_shape;
  branch.w2_shape = *w2_shape;
  branch.w3_shape = *w3_shape;
  return true;
}

bool CanFuseParallelEinsumActivationConcat(
    Ort::ConstNode concat_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>* fusion_nodes) {
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (!IsOnnxOp(concat_node, "Concat") || concat_inputs.size() != 4 ||
      concat_outputs.size() != 1 ||
      fused_node_ids.count(concat_node.GetId()) != 0 ||
      !IsFloatTensorValueInfo(concat_outputs[0])) {
    return false;
  }

  auto concat_axis_attr = GetIntAttribute(concat_node, "axis");
  auto concat_shape = GetTensorShape(concat_outputs[0]);
  if (!concat_axis_attr.has_value() || !concat_shape.has_value() ||
      concat_shape->size() != 3) {
    return false;
  }
  int64_t concat_axis = 0;
  if (!NormalizeAxis(*concat_axis_attr, concat_shape->size(), concat_axis) ||
      concat_axis != 2 || (*concat_shape)[2] != 4) {
    return false;
  }

  std::vector<ParallelEinsumActivationBranchMatch> branches;
  branches.reserve(concat_inputs.size());
  for (size_t i = 0; i < concat_inputs.size(); ++i) {
    ParallelEinsumActivationBranchMatch branch;
    if (!MatchParallelEinsumActivationBranch(concat_node, i, concat_inputs[i],
                                             graph_output_names, fused_node_ids,
                                             branch)) {
      return false;
    }
    if (!branches.empty()) {
      const ParallelEinsumActivationBranchMatch& first = branches.front();
      if (branch.mlp_input_name != first.mlp_input_name ||
          branch.gate_input_name != first.gate_input_name ||
          branch.bias_name != first.bias_name ||
          branch.w1_shape != first.w1_shape ||
          branch.w2_shape != first.w2_shape ||
          branch.w3_shape != first.w3_shape) {
        return false;
      }
    }
    branches.push_back(branch);
  }

  if (fusion_nodes != nullptr) {
    fusion_nodes->clear();
    fusion_nodes->reserve(1 + branches.size() * 7);
    for (const ParallelEinsumActivationBranchMatch& branch : branches) {
      fusion_nodes->push_back(branch.first_einsum);
      fusion_nodes->push_back(branch.first_tanh);
      fusion_nodes->push_back(branch.second_einsum);
      fusion_nodes->push_back(branch.second_tanh);
      fusion_nodes->push_back(branch.third_einsum);
      fusion_nodes->push_back(branch.add);
      fusion_nodes->push_back(branch.mul);
    }
    fusion_nodes->push_back(concat_node);
  }
  return true;
}

bool IsParallelEinsumActivationConcatCandidate(
    Ort::ConstNode concat_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids) {
  return CanFuseParallelEinsumActivationConcat(concat_node, graph_output_names,
                                               fused_node_ids, nullptr);
}

std::vector<std::vector<Ort::ConstNode>> FindParallelEinsumActivationFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode concat_node : all_nodes) {
    if (!IsOnnxOp(concat_node, "Concat") ||
        fused_node_ids.count(concat_node.GetId()) != 0) {
      continue;
    }
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseParallelEinsumActivationConcat(concat_node, graph_output_names,
                                               fused_node_ids, &fusion_nodes)) {
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
