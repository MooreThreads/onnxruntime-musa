// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"

namespace musa_ep {
namespace {

bool CanFuseSilu(Ort::ConstNode mul,
                 const std::unordered_set<std::string>& graph_output_names,
                 const std::unordered_set<size_t>& accepted_node_ids,
                 std::vector<Ort::ConstNode>& fusion) {
  if (!IsOnnxOp(mul, "Mul") || accepted_node_ids.count(mul.GetId())) {
    return false;
  }
  const auto mul_inputs = mul.GetInputs();
  const auto mul_outputs = mul.GetOutputs();
  if (mul_inputs.size() != 2 || mul_outputs.size() != 1) {
    return false;
  }

  for (int sigmoid_input_index = 0; sigmoid_input_index < 2;
       ++sigmoid_input_index) {
    Ort::ConstNode sigmoid =
        mul_inputs[sigmoid_input_index].GetProducerNode().node;
    if (!IsOnnxOp(sigmoid, "Sigmoid") ||
        accepted_node_ids.count(sigmoid.GetId())) {
      continue;
    }
    const auto sigmoid_inputs = sigmoid.GetInputs();
    const auto sigmoid_outputs = sigmoid.GetOutputs();
    if (sigmoid_inputs.size() != 1 || sigmoid_outputs.size() != 1 ||
        !HasOnlyConsumer(sigmoid_outputs[0], mul, sigmoid_input_index)) {
      continue;
    }

    const int x_input_index = 1 - sigmoid_input_index;
    if (Name(sigmoid_inputs[0]) != Name(mul_inputs[x_input_index])) {
      continue;
    }
    if (graph_output_names.count(Name(sigmoid_outputs[0]))) {
      continue;
    }
    ONNXTensorElementDataType elem_type;
    if (!RequireSameElementType(
            {sigmoid_inputs[0], sigmoid_outputs[0], mul_outputs[0]},
            elem_type) ||
        !IsFloatingStorageType(elem_type)) {
      continue;
    }

    fusion = {sigmoid, mul};
    std::sort(fusion.begin(), fusion.end(),
              [](Ort::ConstNode a, Ort::ConstNode b) {
                return a.GetId() < b.GetId();
              });
    return true;
  }
  return false;
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindSiluFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstNode> fusion;
    if (CanFuseSilu(node, graph_output_names, accepted_node_ids, fusion)) {
      fusions.push_back(std::move(fusion));
    }
  }
  return fusions;
}

}  // namespace musa_ep
