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

bool CanFuseTileConcat(
    Ort::ConstNode concat_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(concat_node, "Concat") ||
      accepted_node_ids.count(concat_node.GetId()) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_inputs.empty() || concat_outputs.size() != 1 ||
      graph_output_names.count(Name(concat_outputs[0])) != 0) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  bool found_tile_input = false;
  for (size_t input_idx = 0; input_idx < concat_inputs.size(); ++input_idx) {
    auto producer_it = producers.find(Name(concat_inputs[input_idx]));
    if (producer_it == producers.end() ||
        !IsOnnxOp(producer_it->second, "Tile")) {
      continue;
    }

    Ort::ConstNode tile_node = producer_it->second;
    if (accepted_node_ids.count(tile_node.GetId()) != 0) {
      return false;
    }
    std::vector<Ort::ConstValueInfo> tile_inputs = tile_node.GetInputs();
    std::vector<Ort::ConstValueInfo> tile_outputs = tile_node.GetOutputs();
    if (tile_inputs.size() != 2 || tile_outputs.size() != 1 ||
        graph_output_names.count(Name(tile_outputs[0])) != 0 ||
        !HasOnlyConsumer(tile_outputs[0], concat_node,
                         static_cast<int64_t>(input_idx))) {
      return false;
    }

    found_tile_input = true;
    if (!AddFusionNode(tile_node, accepted_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }

  if (!found_tile_input) {
    return false;
  }
  if (!AddFusionNode(concat_node, accepted_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                     selected_node_ids);
}

std::vector<std::vector<Ort::ConstNode>> FindTileConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);
  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseTileConcat(node, producers, graph_output_names,
                           accepted_node_ids, fusion_nodes)) {
      continue;
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

}  // namespace musa_ep
