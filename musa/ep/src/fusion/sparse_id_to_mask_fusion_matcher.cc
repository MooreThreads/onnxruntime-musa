// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <initializer_list>
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

bool IsCastTo(Ort::ConstNode cast_node, int64_t to) {
  std::optional<int64_t> attr = GetIntAttribute(cast_node, "to");
  return attr.has_value() && *attr == to;
}

bool IsCastToMaskType(Ort::ConstNode cast_node) {
  return IsCastTo(cast_node, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
         IsCastTo(cast_node, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
}

bool IsIntegerTensor(Ort::ConstValueInfo value_info) {
  return IsIntTensorValueInfo(value_info);
}

bool CanBroadcastSparseToBound(const std::vector<int64_t>& sparse_shape,
                               const std::vector<int64_t>& bound_shape) {
  if (sparse_shape.empty() ||
      ShapesEqualOnKnownDims(sparse_shape, bound_shape)) {
    return true;
  }
  if (sparse_shape.size() > bound_shape.size()) {
    return false;
  }
  const size_t offset = bound_shape.size() - sparse_shape.size();
  for (size_t i = 0; i < sparse_shape.size(); ++i) {
    const int64_t sparse_dim = sparse_shape[i];
    const int64_t bound_dim = bound_shape[offset + i];
    if (sparse_dim != 1 && !KnownDimsEqual(sparse_dim, bound_dim)) {
      return false;
    }
  }
  return true;
}

bool HasOnlyConsumers(Ort::ConstValueInfo output,
                      std::initializer_list<Ort::ConstNode> expected_nodes) {
  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
      output.GetConsumers();
  if (consumers.size() != expected_nodes.size()) {
    return false;
  }
  std::unordered_set<size_t> expected_node_ids;
  for (Ort::ConstNode node : expected_nodes) {
    if (!node) {
      return false;
    }
    expected_node_ids.insert(node.GetId());
  }
  for (const auto& consumer : consumers) {
    if (expected_node_ids.count(consumer.node.GetId()) == 0) {
      return false;
    }
  }
  return true;
}

bool CanFuseSparseIdToMask(
    Ort::ConstNode output_cast_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(output_cast_node, "Cast") ||
      accepted_node_ids.count(output_cast_node.GetId()) != 0 ||
      !IsCastToMaskType(output_cast_node)) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> output_cast_inputs =
      output_cast_node.GetInputs();
  std::vector<Ort::ConstValueInfo> output_cast_outputs =
      output_cast_node.GetOutputs();
  if (output_cast_inputs.size() != 1 || output_cast_outputs.size() != 1 ||
      (!IsFloatTensorValueInfo(output_cast_outputs[0]) &&
       !IsIntegerTensor(output_cast_outputs[0]))) {
    return false;
  }

  Ort::ConstNode equal_node = FindProducer(producers, output_cast_inputs[0]);
  if (!IsOnnxOp(equal_node, "Equal") ||
      accepted_node_ids.count(equal_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> equal_inputs = equal_node.GetInputs();
  std::vector<Ort::ConstValueInfo> equal_outputs = equal_node.GetOutputs();
  if (equal_inputs.size() != 2 || equal_outputs.size() != 1 ||
      graph_output_names.count(Name(equal_outputs[0])) != 0 ||
      !HasOnlyConsumer(equal_outputs[0], output_cast_node, 0)) {
    return false;
  }

  Ort::ConstValueInfo dense_input{nullptr};
  Ort::ConstNode add_node{nullptr};
  for (Ort::ConstValueInfo input : equal_inputs) {
    Ort::ConstNode producer = FindProducer(producers, input);
    if (IsOnnxOp(producer, "Add")) {
      add_node = producer;
    } else {
      dense_input = input;
    }
  }
  if (!IsOnnxOp(add_node, "Add") ||
      accepted_node_ids.count(add_node.GetId()) != 0 ||
      dense_input == nullptr || !IsIntegerTensor(dense_input)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  if (add_inputs.size() != 2 || add_outputs.size() != 1 ||
      graph_output_names.count(Name(add_outputs[0])) != 0 ||
      !IsIntegerTensor(add_outputs[0]) ||
      !HasOnlyConsumer(add_outputs[0], equal_node,
                       Name(equal_inputs[0]) == Name(add_outputs[0]) ? 0 : 1)) {
    return false;
  }

  Ort::ConstNode less_equal_node{nullptr};
  Ort::ConstNode not_node{nullptr};
  Ort::ConstNode less_equal_cast_node{nullptr};
  Ort::ConstNode not_cast_node{nullptr};
  Ort::ConstNode default_mul_node{nullptr};
  Ort::ConstNode bound_mul_node{nullptr};
  Ort::ConstValueInfo bound_input{nullptr};
  Ort::ConstValueInfo default_input{nullptr};
  for (Ort::ConstValueInfo add_input : add_inputs) {
    Ort::ConstNode mul_node = FindProducer(producers, add_input);
    if (!IsOnnxOp(mul_node, "Mul") ||
        accepted_node_ids.count(mul_node.GetId()) != 0) {
      return false;
    }
    std::vector<Ort::ConstValueInfo> mul_inputs = mul_node.GetInputs();
    std::vector<Ort::ConstValueInfo> mul_outputs = mul_node.GetOutputs();
    if (mul_inputs.size() != 2 || mul_outputs.size() != 1 ||
        graph_output_names.count(Name(mul_outputs[0])) != 0 ||
        !IsIntegerTensor(mul_outputs[0]) ||
        !HasOnlyConsumer(mul_outputs[0], add_node,
                         Name(add_inputs[0]) == Name(mul_outputs[0]) ? 0 : 1)) {
      return false;
    }

    Ort::ConstValueInfo constant_input{nullptr};
    Ort::ConstValueInfo dynamic_input{nullptr};
    Ort::ConstNode cast_node{nullptr};
    for (Ort::ConstValueInfo mul_input : mul_inputs) {
      if (IsConstantInitializerValueInfo(mul_input)) {
        constant_input = mul_input;
      } else {
        Ort::ConstNode producer = FindProducer(producers, mul_input);
        if (IsOnnxOp(producer, "Cast")) {
          cast_node = producer;
        } else {
          dynamic_input = mul_input;
        }
      }
    }
    if (!IsOnnxOp(cast_node, "Cast") ||
        accepted_node_ids.count(cast_node.GetId()) != 0) {
      return false;
    }
    std::vector<Ort::ConstValueInfo> cast_inputs = cast_node.GetInputs();
    std::vector<Ort::ConstValueInfo> cast_outputs = cast_node.GetOutputs();
    if (cast_inputs.size() != 1 || cast_outputs.size() != 1 ||
        graph_output_names.count(Name(cast_outputs[0])) != 0 ||
        !IsIntegerTensor(cast_outputs[0]) ||
        !HasOnlyConsumer(
            cast_outputs[0], mul_node,
            Name(mul_inputs[0]) == Name(cast_outputs[0]) ? 0 : 1)) {
      return false;
    }

    Ort::ConstNode cast_input_producer =
        FindProducer(producers, cast_inputs[0]);
    if (IsOnnxOp(cast_input_producer, "LessOrEqual")) {
      if (less_equal_node || !IsSmallIntegerInitializer(constant_input)) {
        return false;
      }
      less_equal_node = cast_input_producer;
      less_equal_cast_node = cast_node;
      default_mul_node = mul_node;
      default_input = constant_input;
    } else if (IsOnnxOp(cast_input_producer, "Not")) {
      if (not_node || constant_input != nullptr || dynamic_input == nullptr ||
          !IsIntegerTensor(dynamic_input)) {
        return false;
      }
      not_node = cast_input_producer;
      not_cast_node = cast_node;
      bound_mul_node = mul_node;
      bound_input = dynamic_input;
    } else {
      return false;
    }
  }

  if (!IsOnnxOp(less_equal_node, "LessOrEqual") || !IsOnnxOp(not_node, "Not") ||
      accepted_node_ids.count(less_equal_node.GetId()) != 0 ||
      accepted_node_ids.count(not_node.GetId()) != 0 ||
      bound_input == nullptr || !IsIntegerTensor(bound_input) ||
      !IsSmallIntegerInitializer(default_input)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> not_inputs = not_node.GetInputs();
  std::vector<Ort::ConstValueInfo> not_outputs = not_node.GetOutputs();
  if (not_inputs.size() != 1 || not_outputs.size() != 1 ||
      graph_output_names.count(Name(not_outputs[0])) != 0 ||
      FindProducer(producers, not_inputs[0]).GetId() !=
          less_equal_node.GetId() ||
      !HasOnlyConsumer(not_outputs[0], not_cast_node, 0)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> less_equal_inputs =
      less_equal_node.GetInputs();
  std::vector<Ort::ConstValueInfo> less_equal_outputs =
      less_equal_node.GetOutputs();
  if (less_equal_inputs.size() != 2 || less_equal_outputs.size() != 1 ||
      graph_output_names.count(Name(less_equal_outputs[0])) != 0 ||
      Name(less_equal_inputs[0]) != Name(bound_input) ||
      !IsIntegerTensor(less_equal_inputs[0]) ||
      !IsIntegerTensor(less_equal_inputs[1]) ||
      !HasOnlyConsumers(less_equal_outputs[0],
                        {less_equal_cast_node, not_node})) {
    return false;
  }

  auto dense_shape = GetTensorShape(dense_input);
  auto bound_shape = GetTensorShape(bound_input);
  auto sparse_shape = GetTensorShape(less_equal_inputs[1]);
  auto output_shape = GetTensorShape(output_cast_outputs[0]);
  if (bound_shape.has_value() && sparse_shape.has_value() &&
      !CanBroadcastSparseToBound(*sparse_shape, *bound_shape)) {
    return false;
  }
  if (dense_shape.has_value() && output_shape.has_value() &&
      !ShapesEqualOnKnownDims(*dense_shape, *output_shape)) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  for (Ort::ConstNode node :
       {less_equal_node, not_node, not_cast_node, bound_mul_node,
        less_equal_cast_node, default_mul_node, add_node, equal_node,
        output_cast_node}) {
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

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindSparseIdToMaskFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);
  for (Ort::ConstNode output_cast_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseSparseIdToMask(output_cast_node, producers, graph_output_names,
                               accepted_node_ids, fusion_nodes)) {
      continue;
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

}  // namespace musa_ep
