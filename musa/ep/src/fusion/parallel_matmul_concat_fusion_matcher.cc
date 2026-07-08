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

bool CanFuseParallelMatMulConcat(
    Ort::ConstNode concat_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_inputs.size() < 2 || concat_outputs.size() != 1 ||
      graph_output_names.count(Name(concat_outputs[0])) != 0 ||
      !IsFloatTensorValueInfo(concat_outputs[0])) {
    return false;
  }

  auto concat_axis_attr = GetIntAttribute(concat_node, "axis");
  if (!concat_axis_attr.has_value()) {
    return false;
  }

  std::string common_input_name;
  std::vector<Ort::ConstNode> matmul_nodes;
  std::vector<Ort::ConstNode> unsqueeze_nodes;
  matmul_nodes.reserve(concat_inputs.size());
  unsqueeze_nodes.reserve(concat_inputs.size());
  std::optional<std::vector<int64_t>> first_weight_shape;
  std::optional<std::vector<int64_t>> first_matmul_output_shape;
  int64_t normalized_concat_axis = 0;

  for (size_t i = 0; i < concat_inputs.size(); ++i) {
    Ort::ConstValueInfo concat_input = concat_inputs[i];
    std::vector<Ort::ValueInfoConsumerProducerInfo> unsqueeze_output_consumers =
        concat_input.GetConsumers();
    if (unsqueeze_output_consumers.size() != 1 ||
        unsqueeze_output_consumers[0].node.GetId() != concat_node.GetId() ||
        unsqueeze_output_consumers[0].index != static_cast<int64_t>(i) ||
        graph_output_names.count(Name(concat_input)) != 0 ||
        !IsFloatTensorValueInfo(concat_input)) {
      return false;
    }

    Ort::ValueInfoConsumerProducerInfo producer =
        concat_input.GetProducerNode();
    if (!producer.node || !IsOnnxOp(producer.node, "Unsqueeze") ||
        accepted_node_ids.count(producer.node.GetId()) != 0) {
      return false;
    }
    Ort::ConstNode unsqueeze_node = producer.node;
    std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
        unsqueeze_node.GetInputs();
    std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
        unsqueeze_node.GetOutputs();
    if (unsqueeze_inputs.empty() || unsqueeze_outputs.size() != 1 ||
        Name(unsqueeze_outputs[0]) != Name(concat_input)) {
      return false;
    }

    Ort::ConstValueInfo matmul_output = unsqueeze_inputs[0];
    std::vector<Ort::ValueInfoConsumerProducerInfo> matmul_output_consumers =
        matmul_output.GetConsumers();
    if (matmul_output_consumers.size() != 1 ||
        matmul_output_consumers[0].node.GetId() != unsqueeze_node.GetId() ||
        matmul_output_consumers[0].index != 0 ||
        graph_output_names.count(Name(matmul_output)) != 0 ||
        !IsFloatTensorValueInfo(matmul_output)) {
      return false;
    }

    Ort::ValueInfoConsumerProducerInfo matmul_producer =
        matmul_output.GetProducerNode();
    if (!matmul_producer.node || !IsOnnxOp(matmul_producer.node, "MatMul") ||
        accepted_node_ids.count(matmul_producer.node.GetId()) != 0) {
      return false;
    }
    Ort::ConstNode matmul_node = matmul_producer.node;
    std::vector<Ort::ConstValueInfo> matmul_inputs = matmul_node.GetInputs();
    std::vector<Ort::ConstValueInfo> matmul_outputs = matmul_node.GetOutputs();
    if (matmul_inputs.size() != 2 || matmul_outputs.size() != 1 ||
        Name(matmul_outputs[0]) != Name(matmul_output) ||
        !IsFloatTensorValueInfo(matmul_inputs[0]) ||
        !IsFloatTensorValueInfo(matmul_inputs[1])) {
      return false;
    }

    const std::string input_name = Name(matmul_inputs[0]);
    if (common_input_name.empty()) {
      common_input_name = input_name;
    } else if (common_input_name != input_name) {
      return false;
    }

    auto weight_shape = GetStaticShape(matmul_inputs[1]);
    auto matmul_shape = GetTensorShape(matmul_outputs[0]);
    auto unsqueeze_shape = GetTensorShape(unsqueeze_outputs[0]);
    if (!weight_shape.has_value() || !matmul_shape.has_value() ||
        !unsqueeze_shape.has_value() || weight_shape->size() != 2 ||
        matmul_shape->size() < 2 ||
        unsqueeze_shape->size() != matmul_shape->size() + 1 ||
        !KnownDimsEqual(matmul_shape->back(), (*weight_shape)[1])) {
      return false;
    }

    if (first_weight_shape.has_value()) {
      if (*first_weight_shape != *weight_shape ||
          !ShapesEqualOnKnownDims(*first_matmul_output_shape, *matmul_shape)) {
        return false;
      }
    } else {
      first_weight_shape = weight_shape;
      first_matmul_output_shape = matmul_shape;
    }

    if (!NormalizeAxis(*concat_axis_attr, unsqueeze_shape->size(),
                       normalized_concat_axis) ||
        normalized_concat_axis !=
            static_cast<int64_t>(matmul_shape->size() - 1)) {
      return false;
    }

    auto axes = ReadUnsqueezeAxes(unsqueeze_node);
    int64_t normalized_unsqueeze_axis = 0;
    if (!axes.has_value() || axes->size() != 1 ||
        !NormalizeAxis((*axes)[0], unsqueeze_shape->size(),
                       normalized_unsqueeze_axis) ||
        normalized_unsqueeze_axis != normalized_concat_axis) {
      return false;
    }

    for (size_t dim = 0; dim < unsqueeze_shape->size(); ++dim) {
      if (dim == static_cast<size_t>(normalized_concat_axis)) {
        if ((*unsqueeze_shape)[dim] != 1) {
          return false;
        }
        continue;
      }
      const size_t matmul_dim =
          dim < static_cast<size_t>(normalized_concat_axis) ? dim : dim - 1;
      if (!KnownDimsEqual((*unsqueeze_shape)[dim],
                          (*matmul_shape)[matmul_dim])) {
        return false;
      }
    }

    matmul_nodes.push_back(matmul_node);
    unsqueeze_nodes.push_back(unsqueeze_node);
  }

  auto concat_shape = GetTensorShape(concat_outputs[0]);
  if (!concat_shape.has_value() ||
      concat_shape->size() != first_matmul_output_shape->size() + 1) {
    return false;
  }
  for (size_t dim = 0; dim < concat_shape->size(); ++dim) {
    if (dim == static_cast<size_t>(normalized_concat_axis)) {
      if ((*concat_shape)[dim] != static_cast<int64_t>(concat_inputs.size())) {
        return false;
      }
      continue;
    }
    const size_t matmul_dim =
        dim < static_cast<size_t>(normalized_concat_axis) ? dim : dim - 1;
    if (!KnownDimsEqual((*concat_shape)[dim],
                        (*first_matmul_output_shape)[matmul_dim])) {
      return false;
    }
  }

  fusion_nodes.clear();
  fusion_nodes.reserve(1 + matmul_nodes.size() + unsqueeze_nodes.size());
  for (size_t i = 0; i < matmul_nodes.size(); ++i) {
    fusion_nodes.push_back(matmul_nodes[i]);
    fusion_nodes.push_back(unsqueeze_nodes[i]);
  }
  fusion_nodes.push_back(concat_node);
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindParallelMatMulConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode concat_node : all_nodes) {
    if (!IsOnnxOp(concat_node, "Concat") ||
        accepted_node_ids.count(concat_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseParallelMatMulConcat(concat_node, graph_output_names,
                                     accepted_node_ids, fusion_nodes)) {
      continue;
    }
    fusions.push_back(std::move(fusion_nodes));
  }

  return fusions;
}

}  // namespace musa_ep
