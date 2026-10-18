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
#include "fusion/tile_concat_fusion.h"
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
  return true;
}

// The values a fused node declares as inputs. A Shape source has to be one of
// them: the fused node cannot gain an input that Tile+Concat do not already
// read, so a chain reaching outside would not be recomputable at run time.
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

}  // namespace

bool CanFuseTileConcat(
    Ort::ConstNode concat_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes,
    FusionShapeExprHints* shape_expr_hints) {
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
  // tensor onto the device. Do not absorb the shape-only chain that produces
  // it: the dnn models also read its intermediate values from other branches
  // (MathConcatLog and friends), a fused node owns every value it absorbs, and
  // such a chain would have vetoed the fusion outright. Keep the chain in the
  // graph, leave `repeats` an input of the fused node, and prove here that the
  // value is a pure function of the shapes of the fused inputs. The fused node
  // then recomputes it from those shapes (no device read, no stream sync) and
  // still falls back to reading the tensor if evaluation ever fails.
  //
  // Compile() only receives the Tile+Concat subgraph, so the proof is handed
  // over through `shape_expr_hints`: GetCapability() is the last point that can
  // see the chain. A Tile whose repeats cannot be proven shape-only keeps its
  // standalone kernel and becomes an input of the fused Concat.
  std::unordered_set<std::string> initializer_names;
  std::unordered_map<std::string, ShapeExprConstant> initializer_values;
  std::vector<bool> recomputable(tile_nodes.size(), true);
  for (size_t i = 0; i < tile_nodes.size(); ++i) {
    const std::string repeats = Name(tile_nodes[i].GetInputs()[1]);
    if (producers.find(repeats) == producers.end()) {
      continue;  // initializer or graph input: keep reading it as today
    }
    ShapeChain chain;
    std::unordered_set<std::string> visiting;
    if (!CollectShapeChain(repeats, producers, visiting, chain)) {
      recomputable[i] = false;  // not a shape-only chain
      continue;
    }
    for (const auto& [name, value_info] : chain.leaves) {
      if (initializer_names.count(name) != 0) {
        continue;
      }
      auto constant = ReadShapeExprConstant(value_info);
      if (!constant.has_value()) {
        continue;
      }
      initializer_names.insert(name);
      initializer_values.emplace(name, std::move(*constant));
    }
  }

  // Dropping a Tile shrinks the fusion boundary, which can invalidate another
  // Tile whose Shape source lived on the dropped Tile's input. Rebuild until
  // the set stops shrinking.
  std::vector<Ort::ConstNode> fused_tiles;
  std::unordered_map<std::string, ShapeExprHint> repeats_hints;
  for (;;) {
    fused_tiles.clear();
    for (size_t i = 0; i < tile_nodes.size(); ++i) {
      if (recomputable[i]) {
        fused_tiles.push_back(tile_nodes[i]);
      }
    }
    if (fused_tiles.empty()) {
      return false;
    }

    std::vector<Ort::ConstNode> base_nodes = fused_tiles;
    base_nodes.push_back(concat_node);
    const std::unordered_map<std::string, size_t> base_inputs =
        ExternalInputs(base_nodes);
    std::vector<std::string> input_names(base_inputs.size());
    for (const auto& [name, index] : base_inputs) {
      input_names[index] = name;
    }

    repeats_hints.clear();
    bool dropped = false;
    for (size_t i = 0; i < tile_nodes.size(); ++i) {
      if (!recomputable[i]) {
        continue;
      }
      const std::string repeats = Name(tile_nodes[i].GetInputs()[1]);
      if (producers.find(repeats) == producers.end()) {
        continue;
      }
      ShapeExpr expr = BuildShapeExpr(repeats, producers, base_inputs,
                                      initializer_names, initializer_values);
      if (expr.Empty()) {
        // A Shape source that is not an input of Tile+Concat cannot be
        // recomputed inside the fusion.
        recomputable[i] = false;
        dropped = true;
        continue;
      }
      ShapeExprHint hint;
      hint.source_input_names.reserve(expr.source_input_indices.size());
      for (size_t source_index : expr.source_input_indices) {
        hint.source_input_names.push_back(input_names[source_index]);
      }
      hint.expr = std::move(expr);
      repeats_hints.emplace(repeats, std::move(hint));
    }
    if (!dropped) {
      break;
    }
  }

  // Compile() picks the factory by scanning its detectors in a fixed order, and
  // IsShapeReshapeFusionGraph is a loose presence check (Gather + Concat +
  // Reshape). That used to matter: while the repeats chain was absorbed, the
  // fused body carried the chain's Gather and Reshape next to this fusion's
  // Concat, so such a body was handed to the ShapeReshape factory and the Tile
  // was left unfused. The chain now stays in the graph, the fused body holds
  // only Tile + Concat, and the collision cannot arise.
  const std::vector<std::string> fusion_outputs = {Name(concat_outputs[0])};

  fusion_nodes.clear();
  std::unordered_set<size_t> selected_node_ids;
  for (Ort::ConstNode node : fused_tiles) {
    if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }
  if (!AddFusionNode(concat_node, accepted_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  if (!FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                   selected_node_ids)) {
    return false;
  }

  if (shape_expr_hints != nullptr && !repeats_hints.empty()) {
    std::vector<std::string> boundary_names;
    boundary_names.reserve(fusion_nodes.size() * 2);
    for (const auto& [name, index] : ExternalInputs(fusion_nodes)) {
      boundary_names.push_back(name);
    }
    (*shape_expr_hints)[TileConcatFusionKey(boundary_names, fusion_outputs)] =
        std::move(repeats_hints);
  }
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindTileConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    FusionShapeExprHints* shape_expr_hints) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);
  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseTileConcat(node, producers, graph_output_names,
                           accepted_node_ids, fusion_nodes, shape_expr_hints)) {
      continue;
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

}  // namespace musa_ep
