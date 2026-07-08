// Copyright (c) Microsoft Corporation. All rights reserved.
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

bool CanFuseSplitUnsqueezeConcat(
    Ort::ConstNode reshape_node, Ort::ConstNode split_node,
    const std::vector<Ort::ConstNode>& unsqueeze_nodes,
    Ort::ConstNode concat_node, Ort::ConstNode transpose_node,
    const std::unordered_set<std::string>& graph_output_names) {
  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (split_inputs.empty() || split_inputs.size() > 2 ||
      split_outputs.size() < 2 ||
      unsqueeze_nodes.size() != split_outputs.size() ||
      concat_inputs.size() != split_outputs.size() ||
      concat_outputs.size() != 1) {
    return false;
  }

  Ort::ConstValueInfo packed_value = split_inputs[0];
  Ort::ConstValueInfo source_value = split_inputs[0];
  std::optional<std::vector<int64_t>> reshape_target_shape;
  if (reshape_node) {
    std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
    std::vector<Ort::ConstValueInfo> reshape_outputs =
        reshape_node.GetOutputs();
    if (reshape_inputs.empty() || reshape_outputs.size() != 1 ||
        Name(split_inputs[0]) != Name(reshape_outputs[0]) ||
        graph_output_names.count(Name(reshape_outputs[0])) != 0) {
      return false;
    }
    packed_value = reshape_outputs[0];
    source_value = reshape_inputs[0];
    if (reshape_inputs.size() >= 2) {
      reshape_target_shape = ReadIntInitializerNoLimit(reshape_inputs[1]);
    }
  }

  if (!IsFloatTensorValueInfo(source_value) ||
      !IsFloatTensorValueInfo(packed_value) ||
      !IsFloatTensorValueInfo(concat_outputs[0])) {
    return false;
  }

  const int64_t part_count = static_cast<int64_t>(split_outputs.size());
  int64_t batch = -1;
  int64_t sequence = -1;
  int64_t packed_width = -1;
  int64_t part_width_from_outputs = -1;

  if (reshape_target_shape.has_value()) {
    if (reshape_target_shape->size() != 3) {
      return false;
    }
    if ((*reshape_target_shape)[0] > 0) {
      batch = (*reshape_target_shape)[0];
    }
    if ((*reshape_target_shape)[1] > 0) {
      sequence = (*reshape_target_shape)[1];
    }
    if ((*reshape_target_shape)[2] > 0) {
      packed_width = (*reshape_target_shape)[2];
    }
  }

  auto packed_shape = GetTensorShape(packed_value);
  if (packed_shape.has_value()) {
    if (packed_shape->size() != 3) {
      return false;
    }
    if ((*packed_shape)[0] > 0) {
      if (batch > 0 && batch != (*packed_shape)[0]) {
        return false;
      }
      batch = (*packed_shape)[0];
    }
    if ((*packed_shape)[1] > 0) {
      if (sequence > 0 && sequence != (*packed_shape)[1]) {
        return false;
      }
      sequence = (*packed_shape)[1];
    }
    if ((*packed_shape)[2] > 0) {
      if (packed_width > 0 && packed_width != (*packed_shape)[2]) {
        return false;
      }
      packed_width = (*packed_shape)[2];
    }
  }

  auto first_split_shape = GetTensorShape(split_outputs[0]);
  if (first_split_shape.has_value()) {
    if (first_split_shape->size() != 3) {
      return false;
    }
    if ((*first_split_shape)[0] > 0) {
      if (batch > 0 && batch != (*first_split_shape)[0]) {
        return false;
      }
      batch = (*first_split_shape)[0];
    }
    if ((*first_split_shape)[1] > 0) {
      if (sequence > 0 && sequence != (*first_split_shape)[1]) {
        return false;
      }
      sequence = (*first_split_shape)[1];
    }
    if ((*first_split_shape)[2] > 0) {
      part_width_from_outputs = (*first_split_shape)[2];
    }
  }

  auto split_axis_attr = GetIntAttribute(split_node, "axis");
  int64_t split_axis = 0;
  if (!NormalizeAxis(split_axis_attr.value_or(0), 3, split_axis) ||
      split_axis != 2) {
    return false;
  }

  auto concat_axis_attr = GetIntAttribute(concat_node, "axis");
  int64_t concat_axis = 0;
  if (!concat_axis_attr.has_value() ||
      !NormalizeAxis(*concat_axis_attr, 4, concat_axis) || concat_axis != 0) {
    return false;
  }

  std::vector<int64_t> split_sizes;
  int64_t part_width = part_width_from_outputs;
  if (split_inputs.size() == 2) {
    auto split_initializer = ReadIntInitializerNoLimit(split_inputs[1]);
    if (!split_initializer.has_value() ||
        split_initializer->size() != split_outputs.size()) {
      return false;
    }
    split_sizes = std::move(*split_initializer);
    if (split_sizes.empty() || split_sizes[0] <= 0) {
      return false;
    }
    part_width = split_sizes[0];
  } else if (packed_width > 0) {
    if (packed_width % part_count != 0) {
      return false;
    }
    split_sizes.assign(split_outputs.size(), packed_width / part_count);
    part_width = split_sizes[0];
  } else if (part_width_from_outputs > 0) {
    packed_width = part_width_from_outputs * part_count;
    split_sizes.assign(split_outputs.size(), part_width_from_outputs);
    part_width = part_width_from_outputs;
  }

  if (part_width_from_outputs > 0 && part_width > 0 &&
      part_width != part_width_from_outputs) {
    return false;
  }
  int64_t split_total = 0;
  for (int64_t split_size : split_sizes) {
    if (split_size <= 0 || (part_width > 0 && split_size != part_width)) {
      return false;
    }
    split_total += split_size;
  }
  if (part_width <= 0 && !split_sizes.empty()) {
    part_width = split_sizes[0];
  }
  if (packed_width <= 0 && split_total > 0) {
    packed_width = split_total;
  }
  if (packed_width > 0 && split_total > 0 && split_total != packed_width) {
    return false;
  }

  for (size_t i = 0; i < split_outputs.size(); ++i) {
    Ort::ConstValueInfo split_output = split_outputs[i];
    Ort::ConstNode unsqueeze_node = unsqueeze_nodes[i];
    std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
        unsqueeze_node.GetInputs();
    std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
        unsqueeze_node.GetOutputs();
    if (unsqueeze_inputs.empty() || unsqueeze_outputs.size() != 1 ||
        Name(unsqueeze_inputs[0]) != Name(split_output) ||
        Name(concat_inputs[i]) != Name(unsqueeze_outputs[0]) ||
        graph_output_names.count(Name(split_output)) != 0 ||
        graph_output_names.count(Name(unsqueeze_outputs[0])) != 0 ||
        !IsFloatTensorValueInfo(split_output) ||
        !IsFloatTensorValueInfo(unsqueeze_outputs[0])) {
      return false;
    }

    auto split_shape = GetTensorShape(split_output);
    if (split_shape.has_value()) {
      if (split_shape->size() != 3 ||
          ((*split_shape)[0] > 0 && batch > 0 && (*split_shape)[0] != batch) ||
          ((*split_shape)[1] > 0 && sequence > 0 &&
           (*split_shape)[1] != sequence) ||
          ((*split_shape)[2] > 0 && part_width > 0 &&
           (*split_shape)[2] != part_width)) {
        return false;
      }
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> split_consumers =
        split_output.GetConsumers();
    if (split_consumers.size() != 1 ||
        split_consumers[0].node.GetId() != unsqueeze_node.GetId() ||
        split_consumers[0].index != 0) {
      return false;
    }

    auto axes = ReadUnsqueezeAxes(unsqueeze_node);
    int64_t unsqueeze_axis = 0;
    if (!axes.has_value() || axes->size() != 1 ||
        !NormalizeAxis((*axes)[0], 4, unsqueeze_axis) || unsqueeze_axis != 0) {
      return false;
    }

    auto unsqueeze_shape = GetTensorShape(unsqueeze_outputs[0]);
    if (unsqueeze_shape.has_value()) {
      if (unsqueeze_shape->size() != 4 ||
          ((*unsqueeze_shape)[0] > 0 && (*unsqueeze_shape)[0] != 1) ||
          ((*unsqueeze_shape)[1] > 0 && batch > 0 &&
           (*unsqueeze_shape)[1] != batch) ||
          ((*unsqueeze_shape)[2] > 0 && sequence > 0 &&
           (*unsqueeze_shape)[2] != sequence) ||
          ((*unsqueeze_shape)[3] > 0 && part_width > 0 &&
           (*unsqueeze_shape)[3] != part_width)) {
        return false;
      }
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> unsqueeze_consumers =
        unsqueeze_outputs[0].GetConsumers();
    if (unsqueeze_consumers.size() != 1 ||
        unsqueeze_consumers[0].node.GetId() != concat_node.GetId() ||
        unsqueeze_consumers[0].index != static_cast<int64_t>(i)) {
      return false;
    }
  }

  auto concat_shape = GetTensorShape(concat_outputs[0]);
  if (concat_shape.has_value()) {
    if (concat_shape->size() != 4 ||
        ((*concat_shape)[0] > 0 && (*concat_shape)[0] != part_count) ||
        ((*concat_shape)[1] > 0 && batch > 0 && (*concat_shape)[1] != batch) ||
        ((*concat_shape)[2] > 0 && sequence > 0 &&
         (*concat_shape)[2] != sequence) ||
        ((*concat_shape)[3] > 0 && part_width > 0 &&
         (*concat_shape)[3] != part_width)) {
      return false;
    }
  }

  if (!transpose_node) {
    return true;
  }

  if (graph_output_names.count(Name(concat_outputs[0])) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> transpose_inputs =
      transpose_node.GetInputs();
  std::vector<Ort::ConstValueInfo> transpose_outputs =
      transpose_node.GetOutputs();
  if (transpose_inputs.size() != 1 || transpose_outputs.size() != 1 ||
      Name(transpose_inputs[0]) != Name(concat_outputs[0]) ||
      !IsFloatTensorValueInfo(transpose_outputs[0])) {
    return false;
  }

  auto perm = GetIntsAttribute(transpose_node, "perm");
  if (!perm.has_value() || *perm != std::vector<int64_t>({0, 1, 3, 2})) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> concat_consumers =
      concat_outputs[0].GetConsumers();
  if (concat_consumers.size() != 1 ||
      concat_consumers[0].node.GetId() != transpose_node.GetId() ||
      concat_consumers[0].index != 0) {
    return false;
  }

  auto transpose_shape = GetTensorShape(transpose_outputs[0]);
  if (transpose_shape.has_value()) {
    if (transpose_shape->size() != 4 ||
        ((*transpose_shape)[0] > 0 && (*transpose_shape)[0] != part_count) ||
        ((*transpose_shape)[1] > 0 && batch > 0 &&
         (*transpose_shape)[1] != batch) ||
        ((*transpose_shape)[2] > 0 && part_width > 0 &&
         (*transpose_shape)[2] != part_width) ||
        ((*transpose_shape)[3] > 0 && sequence > 0 &&
         (*transpose_shape)[3] != sequence)) {
      return false;
    }
  }

  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindSplitUnsqueezeConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode split_node : all_nodes) {
    if (!IsOnnxOp(split_node, "Split") ||
        fused_node_ids.count(split_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
    std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
    if (split_inputs.empty() || split_outputs.size() < 2) {
      continue;
    }

    Ort::ConstNode reshape_node{nullptr};
    Ort::ValueInfoConsumerProducerInfo producer =
        split_inputs[0].GetProducerNode();
    if (producer.node && IsOnnxOp(producer.node, "Reshape") &&
        fused_node_ids.count(producer.node.GetId()) == 0) {
      std::vector<Ort::ConstValueInfo> reshape_outputs =
          producer.node.GetOutputs();
      if (reshape_outputs.size() == 1 &&
          reshape_outputs[0].GetConsumers().size() == 1) {
        reshape_node = producer.node;
      }
    }

    std::vector<Ort::ConstNode> unsqueeze_nodes;
    unsqueeze_nodes.reserve(split_outputs.size());
    Ort::ConstNode concat_node{nullptr};
    bool all_outputs_feed_same_concat = true;
    for (Ort::ConstValueInfo split_output : split_outputs) {
      std::vector<Ort::ValueInfoConsumerProducerInfo> split_consumers =
          split_output.GetConsumers();
      if (split_consumers.size() != 1 ||
          !IsOnnxOp(split_consumers[0].node, "Unsqueeze") ||
          fused_node_ids.count(split_consumers[0].node.GetId()) != 0) {
        all_outputs_feed_same_concat = false;
        break;
      }

      Ort::ConstNode unsqueeze_node = split_consumers[0].node;
      std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
          unsqueeze_node.GetOutputs();
      if (unsqueeze_outputs.size() != 1) {
        all_outputs_feed_same_concat = false;
        break;
      }
      std::vector<Ort::ValueInfoConsumerProducerInfo> unsqueeze_consumers =
          unsqueeze_outputs[0].GetConsumers();
      if (unsqueeze_consumers.size() != 1 ||
          !IsOnnxOp(unsqueeze_consumers[0].node, "Concat")) {
        all_outputs_feed_same_concat = false;
        break;
      }
      if (!concat_node) {
        concat_node = unsqueeze_consumers[0].node;
      } else if (concat_node.GetId() != unsqueeze_consumers[0].node.GetId()) {
        all_outputs_feed_same_concat = false;
        break;
      }
      unsqueeze_nodes.push_back(unsqueeze_node);
    }
    if (!all_outputs_feed_same_concat || !concat_node ||
        fused_node_ids.count(concat_node.GetId()) != 0) {
      continue;
    }

    Ort::ConstNode transpose_node{nullptr};
    std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
    if (concat_outputs.size() == 1) {
      std::vector<Ort::ValueInfoConsumerProducerInfo> concat_consumers =
          concat_outputs[0].GetConsumers();
      if (concat_consumers.size() == 1 &&
          IsOnnxOp(concat_consumers[0].node, "Transpose") &&
          fused_node_ids.count(concat_consumers[0].node.GetId()) == 0) {
        transpose_node = concat_consumers[0].node;
      }
    }

    bool fuse_transpose =
        transpose_node && CanFuseSplitUnsqueezeConcat(
                              reshape_node, split_node, unsqueeze_nodes,
                              concat_node, transpose_node, graph_output_names);
    bool fuse_without_transpose = CanFuseSplitUnsqueezeConcat(
        reshape_node, split_node, unsqueeze_nodes, concat_node,
        Ort::ConstNode{nullptr}, graph_output_names);
    if (!fuse_transpose && !fuse_without_transpose) {
      continue;
    }

    std::vector<Ort::ConstNode> fusion_nodes;
    fusion_nodes.reserve(2 + (reshape_node ? 1 : 0) + unsqueeze_nodes.size() +
                         (fuse_transpose ? 1 : 0));
    if (reshape_node) {
      fusion_nodes.push_back(reshape_node);
    }
    fusion_nodes.push_back(split_node);
    fusion_nodes.insert(fusion_nodes.end(), unsqueeze_nodes.begin(),
                        unsqueeze_nodes.end());
    fusion_nodes.push_back(concat_node);
    if (fuse_transpose) {
      fusion_nodes.push_back(transpose_node);
    }

    for (Ort::ConstNode node : fusion_nodes) {
      fused_node_ids.insert(node.GetId());
    }
    fusions.push_back(std::move(fusion_nodes));
  }

  return fusions;
}

}  // namespace musa_ep
