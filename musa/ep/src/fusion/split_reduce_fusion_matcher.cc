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

bool CanFuseSplitReduce(
    Ort::ConstNode split_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(split_node, "Split") ||
      accepted_node_ids.count(split_node.GetId()) != 0 ||
      GetIntAttribute(split_node, "axis").value_or(0) != 1) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  if (split_inputs.size() != 2 || split_outputs.size() < 2 ||
      !IsFloatTensorValueInfo(split_inputs[0]) ||
      std::any_of(split_outputs.begin(), split_outputs.end(),
                  [&graph_output_names](Ort::ConstValueInfo output) {
                    return graph_output_names.count(Name(output)) != 0;
                  })) {
    return false;
  }

  auto input_shape = GetTensorShape(split_inputs[0]);
  if (!input_shape.has_value() || input_shape->size() != 3 ||
      (*input_shape)[1] <= 0 || (*input_shape)[2] <= 0) {
    return false;
  }

  auto split_sizes = ReadSmallIntInitializer(split_inputs[1]);
  if (!split_sizes.has_value() || split_sizes->size() != split_outputs.size()) {
    return false;
  }
  int64_t split_total = 0;
  for (int64_t split_size : *split_sizes) {
    if (split_size <= 0) {
      return false;
    }
    split_total += split_size;
  }
  if (split_total != (*input_shape)[1]) {
    return false;
  }

  fusion_nodes.clear();
  fusion_nodes.push_back(split_node);
  for (Ort::ConstValueInfo split_output : split_outputs) {
    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
        split_output.GetConsumers();
    if (consumers.size() != 1 || consumers[0].index != 0) {
      return false;
    }

    Ort::ConstNode reduce_node = consumers[0].node;
    if (accepted_node_ids.count(reduce_node.GetId()) != 0 ||
        !(IsOnnxOp(reduce_node, "ReduceProd") ||
          IsOnnxOp(reduce_node, "ReduceMean")) ||
        GetIntAttribute(reduce_node, "keepdims").value_or(1) != 0 ||
        !ReduceAxesInputIsAxis1(reduce_node)) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> reduce_outputs = reduce_node.GetOutputs();
    if (reduce_outputs.size() != 1 ||
        !IsFloatTensorValueInfo(reduce_outputs[0])) {
      return false;
    }
    fusion_nodes.push_back(reduce_node);
  }
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindSplitReduceFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode split_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseSplitReduce(split_node, graph_output_names, accepted_node_ids,
                            fusion_nodes)) {
      continue;
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

}  // namespace musa_ep
