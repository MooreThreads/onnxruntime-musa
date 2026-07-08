// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "fusion/fusion_matcher_utils.h"

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

#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {

bool HasSingleConsumerAt(
    Ort::ConstValueInfo value_info, Ort::ConstNode expected_node,
    int64_t expected_index,
    const std::unordered_set<std::string>& graph_output_names) {
  if (graph_output_names.count(Name(value_info)) != 0) {
    return false;
  }
  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
      value_info.GetConsumers();
  return consumers.size() == 1 &&
         consumers[0].node.GetId() == expected_node.GetId() &&
         consumers[0].index == expected_index;
}

bool GetProducer(Ort::ConstValueInfo value_info, Ort::ConstNode& producer) {
  Ort::ValueInfoConsumerProducerInfo info = value_info.GetProducerNode();
  if (!info.node) {
    return false;
  }
  producer = info.node;
  return true;
}

bool IsSmallIntegerInitializer(Ort::ConstValueInfo input) {
  if (!IsSmallInitializer(input)) {
    return false;
  }

  Ort::ConstTypeInfo type_info = input.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  ONNXTensorElementDataType elem_type =
      type_info.GetTensorTypeAndShapeInfo().GetElementType();
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

std::unordered_map<std::string, Ort::ConstNode> BuildProducerMap(
    const std::vector<Ort::ConstNode>& all_nodes) {
  std::unordered_map<std::string, Ort::ConstNode> producers;
  for (Ort::ConstNode node : all_nodes) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      if (output != nullptr) {
        producers.emplace(Name(output), node);
      }
    }
  }
  return producers;
}

Ort::ConstNode FindProducer(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo input) {
  if (input == nullptr) {
    return Ort::ConstNode{nullptr};
  }

  auto it = producers.find(Name(input));
  if (it == producers.end()) {
    return Ort::ConstNode{nullptr};
  }
  return it->second;
}

bool HasOnlyConsumer(Ort::ConstValueInfo output, Ort::ConstNode expected_node,
                     int64_t expected_input_index) {
  if (output == nullptr) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
      output.GetConsumers();
  return consumers.size() == 1 &&
         consumers[0].node.GetId() == expected_node.GetId() &&
         consumers[0].index == expected_input_index;
}

bool AddFusionNode(Ort::ConstNode node,
                   const std::unordered_set<size_t>& accepted_node_ids,
                   std::unordered_set<size_t>& selected_node_ids,
                   std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!node || accepted_node_ids.count(node.GetId()) != 0) {
    return false;
  }
  if (selected_node_ids.insert(node.GetId()).second) {
    fusion_nodes.push_back(node);
  }
  return true;
}

bool PathReachesSelectedNode(
    Ort::ConstNode node, const std::unordered_set<size_t>& selected_node_ids,
    std::unordered_set<size_t>& visited_node_ids) {
  if (!node || !visited_node_ids.insert(node.GetId()).second) {
    return false;
  }
  if (selected_node_ids.count(node.GetId()) != 0) {
    return true;
  }
  for (Ort::ConstValueInfo output : node.GetOutputs()) {
    for (const auto& consumer : output.GetConsumers()) {
      if (PathReachesSelectedNode(consumer.node, selected_node_ids,
                                  visited_node_ids)) {
        return true;
      }
    }
  }
  return false;
}

bool FusionHasNoExternalPathBetweenSelectedNodes(
    const std::vector<Ort::ConstNode>& fusion_nodes,
    const std::unordered_set<size_t>& selected_node_ids) {
  for (Ort::ConstNode node : fusion_nodes) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      for (const auto& consumer : output.GetConsumers()) {
        if (selected_node_ids.count(consumer.node.GetId()) != 0) {
          continue;
        }
        std::unordered_set<size_t> visited_node_ids;
        if (PathReachesSelectedNode(consumer.node, selected_node_ids,
                                    visited_node_ids)) {
          return false;
        }
      }
    }
  }
  return true;
}

bool ReduceAxesInputIsAxis1(Ort::ConstNode reduce_node) {
  std::vector<Ort::ConstValueInfo> inputs = reduce_node.GetInputs();
  if (inputs.size() < 2) {
    return false;
  }

  auto axes = ReadSmallIntInitializer(inputs[1]);
  return axes.has_value() && axes->size() == 1 && (*axes)[0] == 1;
}

bool IsReduceSumOrProd(Ort::ConstNode node) {
  return IsOnnxOp(node, "ReduceSum") || IsOnnxOp(node, "ReduceProd");
}

bool ReduceAxesAreLastDim(Ort::ConstNode reduce_node, size_t rank) {
  std::vector<Ort::ConstValueInfo> inputs = reduce_node.GetInputs();
  if (inputs.size() < 2) {
    return false;
  }
  auto axes = ReadSmallIntInitializer(inputs[1]);
  if (!axes.has_value() || axes->size() != 1) {
    return false;
  }
  int64_t axis = 0;
  return NormalizeAxis((*axes)[0], rank, axis) &&
         axis == static_cast<int64_t>(rank - 1);
}

bool ReduceOutputKeepsLastDim(Ort::ConstValueInfo output,
                              const std::vector<int64_t>& input_shape) {
  auto output_shape = GetTensorShape(output);
  if (!output_shape.has_value()) {
    return true;
  }
  if (output_shape->size() != input_shape.size() || output_shape->empty() ||
      output_shape->back() != 1) {
    return false;
  }
  for (size_t i = 0; i + 1 < input_shape.size(); ++i) {
    if ((*output_shape)[i] > 0 && input_shape[i] > 0 &&
        (*output_shape)[i] != input_shape[i]) {
      return false;
    }
  }
  return true;
}

bool ValueHasOnlyConsumers(Ort::ConstValueInfo value_info,
                           Ort::ConstNode expected_consumer) {
  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
      value_info.GetConsumers();
  if (consumers.empty()) {
    return false;
  }
  for (const auto& consumer : consumers) {
    if (consumer.node.GetId() != expected_consumer.GetId()) {
      return false;
    }
  }
  return true;
}

bool ValueHasExternalConsumerOrGraphOutput(
    Ort::ConstValueInfo value_info, Ort::ConstNode internal_consumer,
    const std::unordered_set<std::string>& graph_output_names) {
  if (graph_output_names.count(Name(value_info)) != 0) {
    return true;
  }
  for (const auto& consumer : value_info.GetConsumers()) {
    if (consumer.node.GetId() != internal_consumer.GetId()) {
      return true;
    }
  }
  return false;
}

std::optional<int64_t> ReadScalarIntInitializer(
    Ort::ConstValueInfo value_info) {
  std::optional<std::vector<int64_t>> values =
      ReadSmallIntInitializer(value_info);
  if (!values.has_value() || values->size() != 1) {
    return std::nullopt;
  }
  return (*values)[0];
}

}  // namespace musa_ep
