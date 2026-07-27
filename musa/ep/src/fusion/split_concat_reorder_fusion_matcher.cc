// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

bool CanFuseSplitConcatReorder(
    Ort::ConstNode reshape_node, Ort::ConstNode split_node,
    Ort::ConstNode concat_node,
    const std::unordered_set<std::string>& graph_output_names) {
  std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reshape_outputs = reshape_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (reshape_inputs.empty() || reshape_outputs.size() != 1 ||
      split_inputs.empty() || split_inputs.size() > 2 ||
      split_outputs.size() < 2 ||
      concat_inputs.size() != split_outputs.size() ||
      concat_outputs.size() != 1 ||
      Name(split_inputs[0]) != Name(reshape_outputs[0])) {
    return false;
  }

  if (graph_output_names.count(Name(reshape_outputs[0])) != 0) {
    return false;
  }

  if (!IsFloatTensorValueInfo(reshape_inputs[0]) ||
      !IsFloatTensorValueInfo(reshape_outputs[0]) ||
      !IsFloatTensorValueInfo(concat_outputs[0])) {
    return false;
  }

  auto reshape_shape = GetTensorShape(reshape_outputs[0]);
  if (!reshape_shape.has_value() || reshape_shape->size() != 3 ||
      (*reshape_shape)[1] <= 0 || (*reshape_shape)[2] <= 0) {
    return false;
  }
  const int64_t batch = (*reshape_shape)[0];
  const int64_t sequence = (*reshape_shape)[1];
  const int64_t packed_width = (*reshape_shape)[2];

  auto split_axis_attr = GetIntAttribute(split_node, "axis");
  int64_t split_axis = 0;
  if (!NormalizeAxis(split_axis_attr.value_or(0), 3, split_axis) ||
      split_axis != 2) {
    return false;
  }

  auto concat_axis_attr = GetIntAttribute(concat_node, "axis");
  int64_t concat_axis = 0;
  if (!concat_axis_attr.has_value() ||
      !NormalizeAxis(*concat_axis_attr, 3, concat_axis) || concat_axis != 0) {
    return false;
  }

  std::vector<int64_t> split_sizes;
  if (split_inputs.size() == 2) {
    auto split_initializer = ReadIntInitializerNoLimit(split_inputs[1]);
    if (!split_initializer.has_value() ||
        split_initializer->size() != split_outputs.size()) {
      return false;
    }
    split_sizes = std::move(*split_initializer);
  } else {
    if (packed_width % static_cast<int64_t>(split_outputs.size()) != 0) {
      return false;
    }
    split_sizes.assign(
        split_outputs.size(),
        packed_width / static_cast<int64_t>(split_outputs.size()));
  }

  if (split_sizes.empty() || split_sizes[0] <= 0) {
    return false;
  }
  const int64_t part_width = split_sizes[0];
  int64_t split_total = 0;
  for (int64_t split_size : split_sizes) {
    if (split_size != part_width) {
      return false;
    }
    split_total += split_size;
  }
  if (split_total != packed_width) {
    return false;
  }

  for (size_t i = 0; i < split_outputs.size(); ++i) {
    Ort::ConstValueInfo split_output = split_outputs[i];
    if (Name(concat_inputs[i]) != Name(split_output) ||
        graph_output_names.count(Name(split_output)) != 0 ||
        !IsFloatTensorValueInfo(split_output)) {
      return false;
    }

    auto split_shape = GetTensorShape(split_output);
    if (!split_shape.has_value() || split_shape->size() != 3 ||
        ((*split_shape)[0] > 0 && batch > 0 && (*split_shape)[0] != batch) ||
        (*split_shape)[1] != sequence || (*split_shape)[2] != part_width) {
      return false;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
        split_output.GetConsumers();
    if (consumers.size() != 1 ||
        consumers[0].node.GetId() != concat_node.GetId() ||
        consumers[0].index != static_cast<int64_t>(i)) {
      return false;
    }
  }

  auto concat_shape = GetTensorShape(concat_outputs[0]);
  if (!concat_shape.has_value() || concat_shape->size() != 3 ||
      ((*concat_shape)[0] > 0 && batch > 0 &&
       (*concat_shape)[0] !=
           batch * static_cast<int64_t>(split_outputs.size())) ||
      (*concat_shape)[1] != sequence || (*concat_shape)[2] != part_width) {
    return false;
  }

  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindSplitConcatReorderFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode split_node : all_nodes) {
    if (!IsOnnxOp(split_node, "Split") ||
        accepted_node_ids.count(split_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
    std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
    if (split_inputs.empty() || split_outputs.size() < 2) {
      continue;
    }

    Ort::ValueInfoConsumerProducerInfo producer =
        split_inputs[0].GetProducerNode();
    if (!producer.node || !IsOnnxOp(producer.node, "Reshape") ||
        accepted_node_ids.count(producer.node.GetId()) != 0) {
      continue;
    }
    Ort::ConstNode reshape_node = producer.node;

    std::vector<Ort::ConstValueInfo> reshape_outputs =
        reshape_node.GetOutputs();
    if (reshape_outputs.size() != 1 ||
        reshape_outputs[0].GetConsumers().size() != 1) {
      continue;
    }

    Ort::ConstNode concat_node{nullptr};
    bool all_outputs_feed_same_concat = true;
    for (Ort::ConstValueInfo split_output : split_outputs) {
      std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
          split_output.GetConsumers();
      if (consumers.size() != 1 || !IsOnnxOp(consumers[0].node, "Concat")) {
        all_outputs_feed_same_concat = false;
        break;
      }
      if (!concat_node) {
        concat_node = consumers[0].node;
      } else if (concat_node.GetId() != consumers[0].node.GetId()) {
        all_outputs_feed_same_concat = false;
        break;
      }
    }
    if (!all_outputs_feed_same_concat || !concat_node ||
        accepted_node_ids.count(concat_node.GetId()) != 0) {
      continue;
    }

    if (!CanFuseSplitConcatReorder(reshape_node, split_node, concat_node,
                                   graph_output_names)) {
      continue;
    }

    fusions.push_back({reshape_node, split_node, concat_node});
  }

  return fusions;
}

}  // namespace musa_ep
