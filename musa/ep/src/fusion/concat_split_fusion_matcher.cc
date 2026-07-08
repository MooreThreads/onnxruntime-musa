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

bool CanFuseConcatSplit(
    Ort::ConstNode concat_node, Ort::ConstNode split_node,
    const std::unordered_set<std::string>& graph_output_names,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  if (concat_inputs.empty() || concat_outputs.size() != 1 ||
      split_inputs.size() < 1 || split_inputs.size() > 2 ||
      split_outputs.empty() ||
      graph_output_names.count(Name(concat_outputs[0])) != 0 ||
      Name(concat_outputs[0]) != Name(split_inputs[0])) {
    return false;
  }

  auto concat_axis_attr = GetIntAttribute(concat_node, "axis");
  auto split_axis_attr = GetIntAttribute(split_node, "axis");
  int64_t concat_axis = 0;
  int64_t split_axis = 0;
  if (!concat_axis_attr.has_value() ||
      !NormalizeAxis(*concat_axis_attr, 2, concat_axis) ||
      !NormalizeAxis(split_axis_attr.value_or(0), 2, split_axis) ||
      concat_axis != 1 || split_axis != 1) {
    return false;
  }

  if (split_inputs.size() != 2) {
    return false;
  }
  auto split_sizes = ReadIntInitializerNoLimit(split_inputs[1]);
  if (!split_sizes.has_value() || split_sizes->size() != split_outputs.size()) {
    return false;
  }

  std::vector<int64_t> concat_widths;
  concat_widths.reserve(concat_inputs.size());
  int64_t total_width = 0;
  for (Ort::ConstValueInfo input : concat_inputs) {
    if (!IsFloatTensorValueInfo(input)) {
      return false;
    }
    auto shape = GetTensorShape(input);
    if (!shape.has_value() || shape->size() != 2 || (*shape)[1] <= 0) {
      return false;
    }
    concat_widths.push_back((*shape)[1]);
    total_width += (*shape)[1];
  }

  int64_t split_total = 0;
  for (int64_t size : *split_sizes) {
    if (size <= 0) {
      return false;
    }
    split_total += size;
  }
  if (split_total != total_width) {
    return false;
  }

  std::unordered_set<std::string> split_output_names;
  std::unordered_map<std::string, int64_t> split_widths;
  for (size_t i = 0; i < split_outputs.size(); ++i) {
    split_output_names.insert(Name(split_outputs[i]));
    split_widths.emplace(Name(split_outputs[i]), (*split_sizes)[i]);
  }

  int64_t split_offset = 0;
  size_t concat_input_index = 0;
  int64_t concat_offset = 0;
  for (int64_t split_size : *split_sizes) {
    while (concat_input_index < concat_widths.size() &&
           split_offset >= concat_offset + concat_widths[concat_input_index]) {
      concat_offset += concat_widths[concat_input_index];
      ++concat_input_index;
    }
    if (concat_input_index >= concat_widths.size()) {
      return false;
    }
    const int64_t local_start = split_offset - concat_offset;
    if (local_start < 0 ||
        local_start + split_size > concat_widths[concat_input_index]) {
      return false;
    }
    split_offset += split_size;
  }

  fusion_nodes = {concat_node, split_node};
  for (Ort::ConstValueInfo split_output : split_outputs) {
    for (const auto& consumer : split_output.GetConsumers()) {
      Ort::ConstNode downstream_node = consumer.node;
      if (IsOnnxOp(downstream_node, "Concat")) {
        auto axis_attr = GetIntAttribute(downstream_node, "axis");
        int64_t downstream_axis = 0;
        if (!axis_attr.has_value() ||
            !NormalizeAxis(*axis_attr, 2, downstream_axis) ||
            downstream_axis != 1) {
          continue;
        }
        bool all_inputs_from_split = true;
        for (Ort::ConstValueInfo input : downstream_node.GetInputs()) {
          if (split_output_names.count(Name(input)) == 0) {
            all_inputs_from_split = false;
            break;
          }
        }
        if (all_inputs_from_split) {
          fusion_nodes.push_back(downstream_node);
        }
      } else if (IsOnnxOp(downstream_node, "Sum")) {
        std::vector<Ort::ConstValueInfo> sum_inputs =
            downstream_node.GetInputs();
        std::vector<Ort::ConstValueInfo> sum_outputs =
            downstream_node.GetOutputs();
        if (sum_inputs.size() < 2 || sum_outputs.size() != 1 ||
            !IsFloatTensorValueInfo(sum_outputs[0])) {
          continue;
        }
        int64_t sum_width = -1;
        bool can_fuse_sum = true;
        for (Ort::ConstValueInfo input : sum_inputs) {
          auto width_it = split_widths.find(Name(input));
          if (width_it == split_widths.end()) {
            can_fuse_sum = false;
            break;
          }
          if (sum_width < 0) {
            sum_width = width_it->second;
          } else if (sum_width != width_it->second) {
            can_fuse_sum = false;
            break;
          }
        }
        if (can_fuse_sum) {
          fusion_nodes.push_back(downstream_node);
        }
      }
    }
  }

  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  fusion_nodes.erase(std::unique(fusion_nodes.begin(), fusion_nodes.end(),
                                 [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
                                   return lhs.GetId() == rhs.GetId();
                                 }),
                     fusion_nodes.end());
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindConcatSplitFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode concat_node : all_nodes) {
    if (!IsOnnxOp(concat_node, "Concat") ||
        fused_node_ids.count(concat_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
    if (concat_outputs.size() != 1) {
      continue;
    }
    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
        concat_outputs[0].GetConsumers();
    if (consumers.size() != 1) {
      continue;
    }

    Ort::ConstNode split_node = consumers[0].node;
    if (!IsOnnxOp(split_node, "Split") ||
        fused_node_ids.count(split_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseConcatSplit(concat_node, split_node, graph_output_names,
                            fusion_nodes)) {
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
