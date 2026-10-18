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
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "fusion/shape_expr.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

// The ops a repeats value may be computed from. Every one of them reads only
// shapes and constants (see shape_expr.cc), so the fused node can recompute the
// value from the shapes of its inputs instead of consuming the tensor.
const std::unordered_set<std::string>& ShapeOnlyOps() {
  static const std::unordered_set<std::string> kOps = {"Shape",
                                                       "Gather",
                                                       "Unsqueeze",
                                                       "Squeeze",
                                                       "Concat",
                                                       "Add",
                                                       "Sub",
                                                       "Mul",
                                                       "Div",
                                                       "Cast",
                                                       "Slice",
                                                       "Reshape",
                                                       "Identity",
                                                       "Mod",
                                                       "Floor",
                                                       "Ceil",
                                                       "ReduceProd",
                                                       "Constant",
                                                       "ConstantOfShape"};
  return kOps;
}

struct ShapeChain {
  std::vector<Ort::ConstNode> nodes;  // inputs precede their consumers
  std::unordered_set<size_t> node_ids;
  // Values the chain reads that no node in the graph produces: initializers
  // (which carry their tensor) or data leaves (which make the chain unusable).
  std::unordered_map<std::string, Ort::ConstValueInfo> leaves;
};

bool CollectShapeChain(
    const std::string& value_name,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    std::unordered_set<std::string>& visiting, ShapeChain& chain) {
  auto producer_it = producers.find(value_name);
  if (producer_it == producers.end()) {
    return true;
  }
  Ort::ConstNode node = producer_it->second;
  if (!node || !IsOnnxDomain(node.GetDomain()) ||
      ShapeOnlyOps().count(node.GetOperatorType()) == 0) {
    return false;
  }
  if (chain.node_ids.count(node.GetId()) != 0) {
    return true;
  }
  if (!visiting.insert(value_name).second) {
    return false;  // cycle: not a shape chain we can unroll
  }
  for (Ort::ConstValueInfo input : node.GetInputs()) {
    if (input == nullptr) {
      continue;
    }
    const std::string name = Name(input);
    if (name.empty()) {
      continue;
    }
    if (producers.find(name) == producers.end()) {
      chain.leaves.emplace(name, input);
      continue;
    }
    if (!CollectShapeChain(name, producers, visiting, chain)) {
      return false;
    }
  }
  visiting.erase(value_name);
  chain.node_ids.insert(node.GetId());
  chain.nodes.push_back(node);
  return true;
}

// The values a fused node would declare as inputs, before any chain is
// absorbed. A Shape source has to be one of them, otherwise absorbing the chain
// would add an input (and possibly a device copy) to the fused node.
std::unordered_map<std::string, size_t> ExternalInputs(
    const std::vector<Ort::ConstNode>& nodes) {
  std::unordered_set<std::string> internal;
  for (Ort::ConstNode node : nodes) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      if (output != nullptr) {
        internal.insert(Name(output));
      }
    }
  }

  std::unordered_map<std::string, size_t> inputs;
  for (Ort::ConstNode node : nodes) {
    for (Ort::ConstValueInfo input : node.GetInputs()) {
      if (input == nullptr) {
        continue;
      }
      const std::string name = Name(input);
      if (name.empty() || internal.count(name) != 0) {
        continue;
      }
      inputs.emplace(name, inputs.size());
    }
  }
  return inputs;
}

bool ChainHasExternalConsumer(
    const ShapeChain& chain, const std::unordered_set<size_t>& fusion_ids,
    const std::unordered_set<std::string>& graph_output_names) {
  std::unordered_set<size_t> allowed = fusion_ids;
  allowed.insert(chain.node_ids.begin(), chain.node_ids.end());
  for (Ort::ConstNode node : chain.nodes) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      if (output == nullptr) {
        continue;
      }
      if (graph_output_names.count(Name(output)) != 0) {
        return true;
      }
      for (const auto& consumer : output.GetConsumers()) {
        if (allowed.count(consumer.node.GetId()) == 0) {
          return true;
        }
      }
    }
  }
  return false;
}

}  // namespace

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

  std::vector<Ort::ConstNode> tile_nodes;
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
    tile_nodes.push_back(tile_node);
  }

  if (tile_nodes.empty()) {
    return false;
  }

  // A fused node cannot declare CPU inputs, so ORT copies a runtime repeats
  // tensor onto the device. Recompute such a repeats value inside the fused
  // node instead: absorb the shape-only chain that produces it, then read only
  // the shapes of the fused inputs at run time (no device read, no sync).
  std::vector<Ort::ConstNode> base_nodes = tile_nodes;
  base_nodes.push_back(concat_node);
  const std::unordered_map<std::string, size_t> base_inputs =
      ExternalInputs(base_nodes);

  std::unordered_set<size_t> fusion_ids;
  for (Ort::ConstNode node : base_nodes) {
    fusion_ids.insert(node.GetId());
  }

  ShapeChain chain;
  std::vector<std::string> chain_roots;
  for (Ort::ConstNode tile_node : tile_nodes) {
    const std::string repeats = Name(tile_node.GetInputs()[1]);
    if (producers.find(repeats) == producers.end()) {
      continue;  // initializer or graph input: keep reading it as today
    }
    std::unordered_set<std::string> visiting;
    if (!CollectShapeChain(repeats, producers, visiting, chain)) {
      return false;
    }
    chain_roots.push_back(repeats);
  }

  if (!chain.nodes.empty() &&
      ChainHasExternalConsumer(chain, fusion_ids, graph_output_names)) {
    return false;
  }

  // Compile() picks the factory by scanning its detectors in a fixed order, and
  // IsShapeReshapeFusionGraph is a loose presence check (Gather + Concat +
  // Reshape). The fused body always carries this fusion's Concat, so a chain
  // that also brings a Gather and a Reshape would be handed to the ShapeReshape
  // factory. Leave such a Tile unfused instead of changing that detector.
  if (!chain.nodes.empty()) {
    bool has_gather = false;
    bool has_reshape = false;
    for (Ort::ConstNode node : chain.nodes) {
      has_gather = has_gather || IsOnnxOp(node, "Gather");
      has_reshape = has_reshape || node.GetOperatorType() == "Reshape";
    }
    if (has_gather && has_reshape) {
      return false;
    }
  }

  if (!chain_roots.empty()) {
    std::unordered_set<std::string> initializer_names;
    std::unordered_map<std::string, ShapeExprConstant> initializer_values;
    for (const auto& [name, value_info] : chain.leaves) {
      auto constant = ReadShapeExprConstant(value_info);
      if (!constant.has_value()) {
        continue;
      }
      initializer_names.insert(name);
      initializer_values.emplace(name, std::move(*constant));
    }
    for (const std::string& repeats : chain_roots) {
      if (BuildShapeExpr(repeats, producers, base_inputs, initializer_names,
                         initializer_values)
              .Empty()) {
        return false;
      }
    }
  }

  fusion_nodes.clear();
  std::unordered_set<size_t> selected_node_ids;
  for (Ort::ConstNode node : base_nodes) {
    if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }
  for (Ort::ConstNode node : chain.nodes) {
    if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
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
