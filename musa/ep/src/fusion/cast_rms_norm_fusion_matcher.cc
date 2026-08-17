// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fusion/fusion_dtype.h"
#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

bool IsCastTo(Ort::ConstNode node, ONNXTensorElementDataType to) {
  return IsOnnxOp(node, "Cast") &&
         GetIntAttribute(node, "to").value_or(-1) == static_cast<int64_t>(to);
}

bool HasElementType(Ort::ConstValueInfo value_info,
                    ONNXTensorElementDataType elem_type) {
  auto actual = GetTensorElementType(value_info);
  return actual.has_value() && *actual == elem_type;
}

bool HasSingleConsumer(Ort::ConstValueInfo value_info,
                       Ort::ConstNode expected_node) {
  auto consumers = value_info.GetConsumers();
  return consumers.size() == 1 &&
         consumers[0].node.GetId() == expected_node.GetId();
}

bool HasOnlyConsumers(Ort::ConstValueInfo value_info,
                      std::initializer_list<Ort::ConstNode> expected_nodes) {
  auto consumers = value_info.GetConsumers();
  if (consumers.size() != expected_nodes.size()) {
    return false;
  }
  std::unordered_set<size_t> expected_node_ids;
  for (Ort::ConstNode node : expected_nodes) {
    expected_node_ids.insert(node.GetId());
  }
  for (const auto& consumer : consumers) {
    if (expected_node_ids.count(consumer.node.GetId()) == 0) {
      return false;
    }
  }
  return true;
}

bool ReadReduceAxes(Ort::ConstNode reduce_node,
                    std::optional<std::vector<int64_t>>& axes) {
  std::vector<Ort::ConstValueInfo> inputs = reduce_node.GetInputs();
  if (inputs.size() >= 2) {
    axes = ReadSmallIntInitializer(inputs[1]);
  } else {
    axes = GetIntsAttribute(reduce_node, "axes");
  }
  return axes.has_value();
}

bool CanFuseCastRmsNorm(
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
      !HasElementType(output_mul_outputs[0],
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)) {
    return false;
  }

  Ort::ConstNode output_cast_node{nullptr};
  Ort::ConstValueInfo gamma_input{nullptr};
  Ort::ConstNode producer = FindProducer(producers, output_mul_inputs[0]);
  if (IsCastTo(producer, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)) {
    output_cast_node = producer;
    gamma_input = output_mul_inputs[1];
  } else {
    producer = FindProducer(producers, output_mul_inputs[1]);
    if (IsCastTo(producer, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)) {
      output_cast_node = producer;
      gamma_input = output_mul_inputs[0];
    }
  }
  if (!output_cast_node ||
      accepted_node_ids.count(output_cast_node.GetId()) != 0 ||
      !HasElementType(gamma_input, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)) {
    return false;
  }

  auto gamma_shape = GetTensorShape(gamma_input);
  if (!gamma_shape.has_value() || gamma_shape->size() != 1 ||
      (*gamma_shape)[0] <= 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> output_cast_inputs =
      output_cast_node.GetInputs();
  std::vector<Ort::ConstValueInfo> output_cast_outputs =
      output_cast_node.GetOutputs();
  if (output_cast_inputs.size() != 1 || output_cast_outputs.size() != 1 ||
      graph_output_names.count(Name(output_cast_outputs[0])) != 0 ||
      !HasElementType(output_cast_inputs[0],
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      !HasElementType(output_cast_outputs[0],
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) ||
      !HasSingleConsumer(output_cast_outputs[0], output_mul_node)) {
    return false;
  }

  Ort::ConstNode div_node = FindProducer(producers, output_cast_inputs[0]);
  if (!IsOnnxOp(div_node, "Div") ||
      accepted_node_ids.count(div_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> div_inputs = div_node.GetInputs();
  std::vector<Ort::ConstValueInfo> div_outputs = div_node.GetOutputs();
  if (div_inputs.size() != 2 || div_outputs.size() != 1 ||
      graph_output_names.count(Name(div_outputs[0])) != 0 ||
      !HasElementType(div_outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      !HasSingleConsumer(div_outputs[0], output_cast_node)) {
    return false;
  }

  Ort::ConstNode input_cast_node = FindProducer(producers, div_inputs[0]);
  if (!IsCastTo(input_cast_node, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      accepted_node_ids.count(input_cast_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> input_cast_inputs =
      input_cast_node.GetInputs();
  std::vector<Ort::ConstValueInfo> input_cast_outputs =
      input_cast_node.GetOutputs();
  if (input_cast_inputs.size() != 1 || input_cast_outputs.size() != 1 ||
      graph_output_names.count(Name(input_cast_outputs[0])) != 0 ||
      !HasElementType(input_cast_inputs[0],
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) ||
      !HasElementType(input_cast_outputs[0],
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    return false;
  }

  auto input_shape = GetTensorShape(input_cast_inputs[0]);
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

  Ort::ConstNode sqrt_node = FindProducer(producers, div_inputs[1]);
  if (!IsOnnxOp(sqrt_node, "Sqrt") ||
      accepted_node_ids.count(sqrt_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> sqrt_inputs = sqrt_node.GetInputs();
  std::vector<Ort::ConstValueInfo> sqrt_outputs = sqrt_node.GetOutputs();
  if (sqrt_inputs.size() != 1 || sqrt_outputs.size() != 1 ||
      graph_output_names.count(Name(sqrt_outputs[0])) != 0 ||
      !HasElementType(sqrt_outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      !HasSingleConsumer(sqrt_outputs[0], div_node)) {
    return false;
  }

  Ort::ConstNode add_node = FindProducer(producers, sqrt_inputs[0]);
  if (!IsOnnxOp(add_node, "Add") ||
      accepted_node_ids.count(add_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  if (add_inputs.size() != 2 || add_outputs.size() != 1 ||
      graph_output_names.count(Name(add_outputs[0])) != 0 ||
      !HasElementType(add_outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      !HasSingleConsumer(add_outputs[0], sqrt_node)) {
    return false;
  }

  Ort::ConstNode reduce_node = FindProducer(producers, add_inputs[0]);
  Ort::ConstValueInfo epsilon_input = add_inputs[1];
  if (!IsOnnxOp(reduce_node, "ReduceMean")) {
    reduce_node = FindProducer(producers, add_inputs[1]);
    epsilon_input = add_inputs[0];
  }
  if (!IsOnnxOp(reduce_node, "ReduceMean") ||
      accepted_node_ids.count(reduce_node.GetId()) != 0 ||
      !HasElementType(epsilon_input, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      !ReadScalarFloatInitializer(epsilon_input).has_value()) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> reduce_inputs = reduce_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reduce_outputs = reduce_node.GetOutputs();
  if (reduce_inputs.empty() || reduce_outputs.size() != 1 ||
      GetIntAttribute(reduce_node, "keepdims").value_or(1) != 1 ||
      graph_output_names.count(Name(reduce_outputs[0])) != 0 ||
      !HasElementType(reduce_outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      !ReduceOutputKeepsLastDim(reduce_outputs[0], *input_shape) ||
      !HasSingleConsumer(reduce_outputs[0], add_node)) {
    return false;
  }

  std::optional<std::vector<int64_t>> axes;
  if (!ReadReduceAxes(reduce_node, axes) || axes->size() != 1) {
    return false;
  }
  int64_t normalized_axis = 0;
  if (!NormalizeAxis((*axes)[0], input_shape->size(), normalized_axis) ||
      normalized_axis != static_cast<int64_t>(input_shape->size() - 1)) {
    return false;
  }

  Ort::ConstNode square_node = FindProducer(producers, reduce_inputs[0]);
  if ((!IsOnnxOp(square_node, "Mul") && !IsOnnxOp(square_node, "Pow")) ||
      accepted_node_ids.count(square_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> square_inputs = square_node.GetInputs();
  std::vector<Ort::ConstValueInfo> square_outputs = square_node.GetOutputs();
  if (square_inputs.size() != 2 || square_outputs.size() != 1 ||
      graph_output_names.count(Name(square_outputs[0])) != 0 ||
      !HasElementType(square_outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      !HasSingleConsumer(square_outputs[0], reduce_node)) {
    return false;
  }
  if (IsOnnxOp(square_node, "Mul")) {
    if (Name(square_inputs[0]) != Name(input_cast_outputs[0]) ||
        Name(square_inputs[1]) != Name(input_cast_outputs[0])) {
      return false;
    }
  } else {
    auto exponent = ReadScalarFloatInitializer(square_inputs[1]);
    if (Name(square_inputs[0]) != Name(input_cast_outputs[0]) ||
        !exponent.has_value() || *exponent != 2.0f) {
      return false;
    }
  }
  if (!HasOnlyConsumers(input_cast_outputs[0], {square_node, div_node})) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  for (Ort::ConstNode node :
       {input_cast_node, square_node, reduce_node, add_node, sqrt_node,
        div_node, output_cast_node, output_mul_node}) {
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

std::vector<std::vector<Ort::ConstNode>> FindCastRmsNormFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);
  for (Ort::ConstNode output_mul_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseCastRmsNorm(output_mul_node, producers, graph_output_names,
                            accepted_node_ids, fusion_nodes)) {
      continue;
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

}  // namespace musa_ep
