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

bool CanFuseShapeReshapeFromGather(
    Ort::ConstNode gather_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::unordered_set<size_t>& group_node_ids) {
  if (!IsOnnxOp(gather_node, "Gather") ||
      accepted_node_ids.count(gather_node.GetId()) != 0 ||
      GetIntAttribute(gather_node, "axis").value_or(0) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gather_outputs = gather_node.GetOutputs();
  if (gather_inputs.size() != 2 || gather_outputs.size() != 1 ||
      graph_output_names.count(Name(gather_outputs[0])) != 0 ||
      !IsSmallIntegerInitializer(gather_inputs[1])) {
    return false;
  }

  Ort::ConstNode cast_node = FindProducer(producers, gather_inputs[0]);
  if (!cast_node || !IsOnnxOp(cast_node, "Cast") ||
      accepted_node_ids.count(cast_node.GetId()) != 0) {
    return false;
  }
  auto cast_to = GetIntAttribute(cast_node, "to");
  if (!cast_to.has_value() ||
      (*cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
       *cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> cast_inputs = cast_node.GetInputs();
  std::vector<Ort::ConstValueInfo> cast_outputs = cast_node.GetOutputs();
  if (cast_inputs.size() != 1 || cast_outputs.size() != 1 ||
      graph_output_names.count(Name(cast_outputs[0])) != 0 ||
      !HasOnlyConsumer(cast_outputs[0], gather_node, 0)) {
    return false;
  }

  Ort::ConstNode shape_input_producer = FindProducer(producers, cast_inputs[0]);
  if (!shape_input_producer) {
    return false;
  }

  Ort::ConstNode shape_node{nullptr};
  Ort::ConstNode pre_gather_node{nullptr};
  bool include_shape_node = false;
  if (IsOnnxOp(shape_input_producer, "Shape")) {
    shape_node = shape_input_producer;
    if (accepted_node_ids.count(shape_node.GetId()) != 0) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
    std::vector<Ort::ConstValueInfo> shape_outputs = shape_node.GetOutputs();
    if (shape_inputs.size() != 1 || shape_outputs.size() != 1 ||
        graph_output_names.count(Name(shape_outputs[0])) != 0) {
      return false;
    }
    include_shape_node = HasOnlyConsumer(shape_outputs[0], cast_node, 0);
  } else if (IsOnnxOp(shape_input_producer, "Gather")) {
    pre_gather_node = shape_input_producer;
    if (accepted_node_ids.count(pre_gather_node.GetId()) != 0 ||
        GetIntAttribute(pre_gather_node, "axis").value_or(0) != 0) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> pre_gather_inputs =
        pre_gather_node.GetInputs();
    std::vector<Ort::ConstValueInfo> pre_gather_outputs =
        pre_gather_node.GetOutputs();
    if (pre_gather_inputs.size() != 2 || pre_gather_outputs.size() != 1 ||
        graph_output_names.count(Name(pre_gather_outputs[0])) != 0 ||
        !IsSmallIntegerInitializer(pre_gather_inputs[1]) ||
        !HasOnlyConsumer(pre_gather_outputs[0], cast_node, 0)) {
      return false;
    }

    shape_node = FindProducer(producers, pre_gather_inputs[0]);
    if (!shape_node || !IsOnnxOp(shape_node, "Shape") ||
        accepted_node_ids.count(shape_node.GetId()) != 0) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
    std::vector<Ort::ConstValueInfo> shape_outputs = shape_node.GetOutputs();
    if (shape_inputs.size() != 1 || shape_outputs.size() != 1 ||
        graph_output_names.count(Name(shape_outputs[0])) != 0) {
      return false;
    }
    include_shape_node = HasOnlyConsumer(shape_outputs[0], pre_gather_node, 0);
  } else {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> gather_consumers =
      gather_outputs[0].GetConsumers();
  if (gather_consumers.empty()) {
    return false;
  }

  group_node_ids.clear();
  if (include_shape_node) {
    group_node_ids.insert(shape_node.GetId());
  }
  if (pre_gather_node) {
    group_node_ids.insert(pre_gather_node.GetId());
  }
  group_node_ids.insert(cast_node.GetId());
  group_node_ids.insert(gather_node.GetId());

  for (const auto& gather_consumer : gather_consumers) {
    Ort::ConstNode concat_node = gather_consumer.node;
    if (!IsOnnxOp(concat_node, "Concat") ||
        accepted_node_ids.count(concat_node.GetId()) != 0 ||
        GetIntAttribute(concat_node, "axis").value_or(0) != 0) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
    std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
    if (concat_inputs.size() < 2 || concat_outputs.size() != 1 ||
        graph_output_names.count(Name(concat_outputs[0])) != 0) {
      return false;
    }

    int gather_input_count = 0;
    for (Ort::ConstValueInfo concat_input : concat_inputs) {
      if (Name(concat_input) == Name(gather_outputs[0])) {
        ++gather_input_count;
      } else if (!IsSmallIntegerInitializer(concat_input)) {
        return false;
      }
    }
    if (gather_input_count != 1) {
      return false;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> concat_consumers =
        concat_outputs[0].GetConsumers();
    if (concat_consumers.size() != 1 || concat_consumers[0].index != 0) {
      return false;
    }

    Ort::ConstNode final_cast_node = concat_consumers[0].node;
    if (!IsOnnxOp(final_cast_node, "Cast") ||
        accepted_node_ids.count(final_cast_node.GetId()) != 0 ||
        GetIntAttribute(final_cast_node, "to").value_or(-1) !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> final_cast_outputs =
        final_cast_node.GetOutputs();
    if (final_cast_outputs.size() != 1 ||
        graph_output_names.count(Name(final_cast_outputs[0])) != 0) {
      return false;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> reshape_consumers =
        final_cast_outputs[0].GetConsumers();
    if (reshape_consumers.empty()) {
      return false;
    }

    group_node_ids.insert(concat_node.GetId());
    group_node_ids.insert(final_cast_node.GetId());

    for (const auto& reshape_consumer : reshape_consumers) {
      Ort::ConstNode reshape_node = reshape_consumer.node;
      if (reshape_consumer.index != 1 || !IsOnnxOp(reshape_node, "Reshape") ||
          accepted_node_ids.count(reshape_node.GetId()) != 0) {
        return false;
      }

      std::vector<Ort::ConstValueInfo> reshape_inputs =
          reshape_node.GetInputs();
      std::vector<Ort::ConstValueInfo> reshape_outputs =
          reshape_node.GetOutputs();
      if (reshape_inputs.size() != 2 || reshape_outputs.size() != 1 ||
          reshape_inputs[0].IsConstantInitializer()) {
        return false;
      }
      group_node_ids.insert(reshape_node.GetId());
    }
  }

  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindShapeReshapeFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);

  for (Ort::ConstNode node : all_nodes) {
    std::unordered_set<size_t> group_node_ids;
    if (!CanFuseShapeReshapeFromGather(node, producers, graph_output_names,
                                       accepted_node_ids, group_node_ids)) {
      continue;
    }

    std::vector<Ort::ConstNode> fusion_nodes;
    for (Ort::ConstNode ordered_node : all_nodes) {
      if (group_node_ids.count(ordered_node.GetId()) != 0) {
        fusion_nodes.push_back(ordered_node);
      }
    }
    if (!fusion_nodes.empty()) {
      fusions.push_back(std::move(fusion_nodes));
    }
  }

  return fusions;
}

}  // namespace musa_ep
