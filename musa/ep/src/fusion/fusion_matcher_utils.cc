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

bool HasCastBoundary(const std::vector<Ort::ConstNode>& fusion_nodes) {
  std::unordered_set<size_t> selected_node_ids;
  selected_node_ids.reserve(fusion_nodes.size());
  for (Ort::ConstNode node : fusion_nodes) {
    if (node) {
      selected_node_ids.insert(node.GetId());
    }
  }

  for (Ort::ConstNode node : fusion_nodes) {
    if (!node) {
      continue;
    }
    for (Ort::ConstValueInfo input : node.GetInputs()) {
      Ort::ConstNode producer{nullptr};
      if (GetProducer(input, producer) &&
          selected_node_ids.count(producer.GetId()) == 0 &&
          IsOnnxOp(producer, "Cast")) {
        return true;
      }
    }
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      for (const auto& consumer : output.GetConsumers()) {
        if (selected_node_ids.count(consumer.node.GetId()) == 0 &&
            IsOnnxOp(consumer.node, "Cast")) {
          return true;
        }
      }
    }
  }
  return false;
}

// Ops whose output shape is a pure function of their input shapes. A Shape
// node reading their output can therefore be keyed on the shapes of the fused
// inputs that feed them, without the fused node taking a new input.
bool IsShapePreservingDataOp(const std::string& op) {
  return op == "Transpose" || op == "Cast" || op == "Identity";
}

// Walk from `value_name` through shape-preserving producers to the fused
// inputs that determine its shape. Returns false when the walk leaves the
// fused boundary through anything whose output shape is data-dependent.
bool CollectShapePreservingSources(
    const std::string& value_name,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_map<std::string, size_t>& input_indices,
    const std::unordered_set<std::string>& initializer_names,
    std::unordered_set<size_t>* sources) {
  // A value can also be pinned down from the other side: when another
  // Transpose reads the *same* value and produces one of the fusion's inputs,
  // that input's shape is this value's shape under a fixed permutation (every
  // valid ONNX Transpose perm is invertible), so it can key the cache too.
  std::unordered_map<std::string, std::vector<Ort::ConstNode>> consumers;
  for (const auto& [output, producer] : producers) {
    for (Ort::ConstValueInfo input : producer.GetInputs()) {
      const std::string input_name = input.GetName();
      if (!input_name.empty()) {
        consumers[input_name].push_back(producer);
      }
    }
  }

  std::vector<std::string> pending{value_name};
  std::unordered_set<std::string> visited;
  while (!pending.empty()) {
    const std::string name = std::move(pending.back());
    pending.pop_back();
    if (!visited.insert(name).second) {
      continue;
    }
    auto index_it = input_indices.find(name);
    if (index_it != input_indices.end()) {
      sources->insert(index_it->second);
      continue;
    }
    if (initializer_names.count(name) != 0) {
      continue;  // a constant's shape is static and needs no runtime key
    }
    bool keyed_from_consumer = false;
    auto consumer_it = consumers.find(name);
    if (consumer_it != consumers.end()) {
      for (Ort::ConstNode consumer : consumer_it->second) {
        if (!IsOnnxOp(consumer, "Transpose") ||
            consumer.GetInputs().size() != 1) {
          continue;
        }
        for (Ort::ConstValueInfo output : consumer.GetOutputs()) {
          auto fused_it = input_indices.find(Name(output));
          if (fused_it != input_indices.end()) {
            sources->insert(fused_it->second);
            keyed_from_consumer = true;
          }
        }
        if (keyed_from_consumer) {
          break;
        }
      }
    }
    if (keyed_from_consumer) {
      continue;
    }
    auto producer_it = producers.find(name);
    if (producer_it == producers.end()) {
      return false;
    }
    Ort::ConstNode node = producer_it->second;
    if (!IsShapePreservingDataOp(node.GetOperatorType())) {
      return false;
    }
    for (Ort::ConstValueInfo input : node.GetInputs()) {
      const std::string input_name = input.GetName();
      if (!input_name.empty()) {
        pending.push_back(input_name);
      }
    }
  }
  return true;
}

std::string FusionBoundaryKey(const std::vector<std::string>& input_names,
                              const std::vector<std::string>& output_names) {
  auto normalize = [](std::vector<std::string> names) {
    names.erase(
        std::remove_if(names.begin(), names.end(),
                       [](const std::string& name) { return name.empty(); }),
        names.end());
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    std::string key;
    for (const std::string& name : names) {
      key += name;
      key += '\x1e';
    }
    return key;
  };
  return normalize(input_names) + "|" + normalize(output_names);
}

bool ValueDependsOnlyOnInputShapes(
    const std::string& value_name,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_map<std::string, size_t>& input_indices,
    const std::unordered_set<std::string>& initializer_names,
    std::vector<size_t>* source_indices) {
  // Ops that only reshape or combine shape values. Reshape is included because
  // its output shape is fixed by its (constant) shape input; a data dependency
  // would surface as a non-initializer leaf and be rejected below.
  static const std::unordered_set<std::string> kShapeOnlyOps = {
      "Gather",   "Unsqueeze", "Squeeze",   "Concat",  "Add",
      "Sub",      "Mul",       "Div",       "Cast",    "Slice",
      "Identity", "Mod",       "Floor",     "Ceil",    "Reshape",
      "ReduceProd", "Constant", "ConstantOfShape"};
  std::vector<std::string> pending{value_name};
  std::unordered_set<std::string> visited;
  std::unordered_set<size_t> sources;
  while (!pending.empty()) {
    const std::string name = std::move(pending.back());
    pending.pop_back();
    if (!visited.insert(name).second) {
      continue;
    }
    auto producer_it = producers.find(name);
    if (producer_it == producers.end()) {
      // A leaf must be a constant. A graph input or fused data input used as a
      // value would make the result data-dependent.
      if (initializer_names.count(name) == 0) {
        return false;
      }
      continue;
    }
    Ort::ConstNode node = producer_it->second;
    const std::string op = node.GetOperatorType();
    std::vector<Ort::ConstValueInfo> node_inputs = node.GetInputs();
    if (op == "Shape") {
      if (node_inputs.size() != 1) {
        return false;
      }
      const std::string source = node_inputs[0].GetName();
      auto index_it = input_indices.find(source);
      if (index_it == input_indices.end()) {
        // Shape of a constant is constant and needs no runtime key. Otherwise
        // follow shape-preserving producers (a Transpose kept only for its
        // shape, for example) back to the fused inputs that determine it.
        if (initializer_names.count(source) == 0 &&
            !CollectShapePreservingSources(source, producers, input_indices,
                                           initializer_names, &sources)) {
          return false;
        }
      } else {
        sources.insert(index_it->second);
      }
      continue;
    }
    if (kShapeOnlyOps.count(op) == 0) {
      return false;
    }
    for (Ort::ConstValueInfo input : node_inputs) {
      const std::string input_name = input.GetName();
      if (!input_name.empty()) {
        pending.push_back(input_name);
      }
    }
  }
  source_indices->assign(sources.begin(), sources.end());
  std::sort(source_indices->begin(), source_indices->end());
  return true;
}

}  // namespace musa_ep
