// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

constexpr size_t kSwiGluMaxInputRank = 8;

bool IsSwiGluStorageType(ONNXTensorElementDataType elem_type) {
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
}

bool HasExactlyConsumers(Ort::ConstValueInfo value, Ort::ConstNode first,
                         Ort::ConstNode second) {
  const auto consumers = value.GetConsumers();
  if (consumers.size() != 2) {
    return false;
  }
  bool saw_first = false;
  bool saw_second = false;
  for (const auto& consumer : consumers) {
    saw_first |= consumer.node.GetId() == first.GetId();
    saw_second |= consumer.node.GetId() == second.GetId();
  }
  return saw_first && saw_second;
}

bool MatchSiluMul(Ort::ConstNode gate_mul,
                  const std::unordered_set<std::string>& graph_output_names,
                  const std::unordered_set<size_t>& accepted_node_ids,
                  Ort::ConstNode& sigmoid, Ort::ConstValueInfo& gate) {
  if (!IsOnnxOp(gate_mul, "Mul") ||
      accepted_node_ids.count(gate_mul.GetId()) != 0) {
    return false;
  }
  const auto mul_inputs = gate_mul.GetInputs();
  const auto mul_outputs = gate_mul.GetOutputs();
  if (mul_inputs.size() != 2 || mul_outputs.size() != 1 ||
      graph_output_names.count(Name(mul_outputs[0])) != 0) {
    return false;
  }

  for (size_t sigmoid_index = 0; sigmoid_index < 2; ++sigmoid_index) {
    Ort::ConstNode candidate = mul_inputs[sigmoid_index].GetProducerNode().node;
    if (!IsOnnxOp(candidate, "Sigmoid") ||
        accepted_node_ids.count(candidate.GetId()) != 0) {
      continue;
    }
    const auto sigmoid_inputs = candidate.GetInputs();
    const auto sigmoid_outputs = candidate.GetOutputs();
    if (sigmoid_inputs.size() != 1 || sigmoid_outputs.size() != 1 ||
        Name(sigmoid_inputs[0]) != Name(mul_inputs[1 - sigmoid_index]) ||
        graph_output_names.count(Name(sigmoid_outputs[0])) != 0 ||
        !HasOnlyConsumer(sigmoid_outputs[0], gate_mul,
                         static_cast<int64_t>(sigmoid_index))) {
      continue;
    }
    sigmoid = candidate;
    gate = sigmoid_inputs[0];
    return true;
  }
  return false;
}

bool MatchProjection(Ort::ConstValueInfo value,
                     const std::unordered_set<size_t>& accepted_node_ids,
                     Ort::ConstNode& matmul, Ort::ConstValueInfo& input,
                     Ort::ConstValueInfo& weight) {
  matmul = value.GetProducerNode().node;
  if (!IsOnnxOp(matmul, "MatMul") ||
      accepted_node_ids.count(matmul.GetId()) != 0) {
    return false;
  }
  const auto inputs = matmul.GetInputs();
  const auto outputs = matmul.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1 ||
      Name(outputs[0]) != Name(value) || !inputs[1].IsConstantInitializer()) {
    return false;
  }
  const auto weight_shape = GetStaticShape(inputs[1]);
  if (!weight_shape.has_value() || weight_shape->size() != 2 ||
      (*weight_shape)[0] <= 0 || (*weight_shape)[1] <= 0) {
    return false;
  }
  input = inputs[0];
  weight = inputs[1];
  return true;
}

bool CanFuseSwiGlu(Ort::ConstNode output_mul,
                   const std::unordered_set<std::string>& graph_output_names,
                   const std::unordered_set<size_t>& accepted_node_ids,
                   std::vector<Ort::ConstNode>& fusion) {
  if (!IsOnnxOp(output_mul, "Mul") ||
      accepted_node_ids.count(output_mul.GetId()) != 0) {
    return false;
  }
  const auto output_mul_inputs = output_mul.GetInputs();
  const auto output_mul_outputs = output_mul.GetOutputs();
  if (output_mul_inputs.size() != 2 || output_mul_outputs.size() != 1) {
    return false;
  }

  for (size_t gate_mul_index = 0; gate_mul_index < 2; ++gate_mul_index) {
    Ort::ConstNode gate_mul =
        output_mul_inputs[gate_mul_index].GetProducerNode().node;
    Ort::ConstNode sigmoid{nullptr};
    Ort::ConstValueInfo gate{nullptr};
    if (!MatchSiluMul(gate_mul, graph_output_names, accepted_node_ids, sigmoid,
                      gate)) {
      continue;
    }
    const auto gate_mul_outputs = gate_mul.GetOutputs();
    if (!HasOnlyConsumer(gate_mul_outputs[0], output_mul,
                         static_cast<int64_t>(gate_mul_index))) {
      continue;
    }

    Ort::ConstValueInfo up = output_mul_inputs[1 - gate_mul_index];
    if (graph_output_names.count(Name(gate)) != 0 ||
        graph_output_names.count(Name(up)) != 0 ||
        !HasExactlyConsumers(gate, sigmoid, gate_mul) ||
        !HasOnlyConsumer(up, output_mul,
                         static_cast<int64_t>(1 - gate_mul_index))) {
      continue;
    }

    Ort::ConstNode gate_matmul{nullptr};
    Ort::ConstNode up_matmul{nullptr};
    Ort::ConstValueInfo gate_input{nullptr};
    Ort::ConstValueInfo up_input{nullptr};
    Ort::ConstValueInfo gate_weight{nullptr};
    Ort::ConstValueInfo up_weight{nullptr};
    if (!MatchProjection(gate, accepted_node_ids, gate_matmul, gate_input,
                         gate_weight) ||
        !MatchProjection(up, accepted_node_ids, up_matmul, up_input,
                         up_weight) ||
        gate_matmul.GetId() == up_matmul.GetId() ||
        Name(gate_input) != Name(up_input)) {
      continue;
    }

    const auto gate_weight_shape = GetStaticShape(gate_weight);
    const auto up_weight_shape = GetStaticShape(up_weight);
    if (gate_weight_shape != up_weight_shape) {
      continue;
    }

    const auto input_shape = GetTensorShape(gate_input);
    if (!input_shape.has_value() || input_shape->size() < 2 ||
        input_shape->size() > kSwiGluMaxInputRank ||
        (input_shape->back() != -1 &&
         input_shape->back() != (*gate_weight_shape)[0])) {
      continue;
    }

    ONNXTensorElementDataType elem_type;
    if (!RequireSameElementType({gate_input, gate_weight, up_weight, gate, up,
                                 sigmoid.GetOutputs()[0], gate_mul_outputs[0],
                                 output_mul_outputs[0]},
                                elem_type) ||
        !IsSwiGluStorageType(elem_type)) {
      continue;
    }

    fusion = {gate_matmul, up_matmul, sigmoid, gate_mul, output_mul};
    std::sort(fusion.begin(), fusion.end(),
              [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
                return lhs.GetId() < rhs.GetId();
              });
    return true;
  }
  return false;
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindSwiGluFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstNode> fusion;
    if (CanFuseSwiGlu(node, graph_output_names, accepted_node_ids, fusion)) {
      fusions.push_back(std::move(fusion));
    }
  }
  return fusions;
}

}  // namespace musa_ep
