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
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

bool CastTo(Ort::ConstNode cast_node, int64_t expected_to) {
  return GetIntAttribute(cast_node, "to").value_or(-1) == expected_to;
}

bool CanFuseBucketizeGather(
    Ort::ConstNode squeeze_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(squeeze_node, "Squeeze") ||
      accepted_node_ids.count(squeeze_node.GetId()) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> squeeze_inputs = squeeze_node.GetInputs();
  std::vector<Ort::ConstValueInfo> squeeze_outputs = squeeze_node.GetOutputs();
  if (squeeze_inputs.size() != 2 || squeeze_outputs.size() != 1 ||
      !IsFloatTensorValueInfo(squeeze_outputs[0])) {
    return false;
  }
  auto squeeze_axes = ReadSmallIntInitializer(squeeze_inputs[1]);
  if (!squeeze_axes.has_value() || squeeze_axes->size() != 1) {
    return false;
  }

  auto gather_it = producers.find(Name(squeeze_inputs[0]));
  if (gather_it == producers.end() || !IsOnnxOp(gather_it->second, "Gather")) {
    return false;
  }
  Ort::ConstNode gather_node = gather_it->second;
  if (accepted_node_ids.count(gather_node.GetId()) != 0 ||
      GetIntAttribute(gather_node, "axis").value_or(0) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gather_outputs = gather_node.GetOutputs();
  if (gather_inputs.size() != 2 || gather_outputs.size() != 1 ||
      graph_output_names.count(Name(gather_outputs[0])) != 0 ||
      !HasOnlyConsumer(gather_outputs[0], squeeze_node, 0) ||
      !IsFloatTensorValueInfo(gather_inputs[0]) ||
      !IsFloatTensorValueInfo(gather_outputs[0])) {
    return false;
  }

  auto table_shape = GetTensorShape(gather_inputs[0]);
  auto gather_output_shape = GetTensorShape(gather_outputs[0]);
  if (!table_shape.has_value() || table_shape->empty() ||
      (*table_shape)[0] <= 0 || !gather_output_shape.has_value()) {
    return false;
  }
  int64_t squeeze_axis = 0;
  if (!NormalizeAxis((*squeeze_axes)[0], gather_output_shape->size(),
                     squeeze_axis) ||
      (*gather_output_shape)[static_cast<size_t>(squeeze_axis)] != 1) {
    return false;
  }

  auto final_mul_it = producers.find(Name(gather_inputs[1]));
  if (final_mul_it == producers.end() ||
      !IsOnnxOp(final_mul_it->second, "Mul")) {
    return false;
  }
  Ort::ConstNode final_mul = final_mul_it->second;
  if (accepted_node_ids.count(final_mul.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> final_mul_inputs = final_mul.GetInputs();
  std::vector<Ort::ConstValueInfo> final_mul_outputs = final_mul.GetOutputs();
  if (final_mul_inputs.size() != 2 || final_mul_outputs.size() != 1 ||
      graph_output_names.count(Name(final_mul_outputs[0])) != 0 ||
      !HasOnlyConsumer(final_mul_outputs[0], gather_node, 1) ||
      !IsIntTensorValueInfo(final_mul_outputs[0])) {
    return false;
  }

  Ort::ConstNode add_node{nullptr};
  Ort::ConstNode mask_cast_node{nullptr};
  for (Ort::ConstValueInfo input : final_mul_inputs) {
    auto it = producers.find(Name(input));
    if (it != producers.end() && IsOnnxOp(it->second, "Add")) {
      add_node = it->second;
    } else if (it != producers.end() && IsOnnxOp(it->second, "Cast")) {
      mask_cast_node = it->second;
    }
  }
  if (!add_node || !mask_cast_node ||
      accepted_node_ids.count(add_node.GetId()) != 0 ||
      accepted_node_ids.count(mask_cast_node.GetId()) != 0 ||
      !CastTo(mask_cast_node, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  if (add_inputs.size() != 2 || add_outputs.size() != 1 ||
      graph_output_names.count(Name(add_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(add_outputs[0], final_mul) ||
      !IsIntTensorValueInfo(add_outputs[0])) {
    return false;
  }
  Ort::ConstNode sub_node{nullptr};
  Ort::ConstValueInfo offset_input{nullptr};
  for (Ort::ConstValueInfo input : add_inputs) {
    auto it = producers.find(Name(input));
    if (it != producers.end() && IsOnnxOp(it->second, "Sub")) {
      sub_node = it->second;
    } else {
      offset_input = input;
    }
  }
  std::optional<int64_t> offset = ReadScalarIntInitializer(offset_input);
  if (!sub_node || accepted_node_ids.count(sub_node.GetId()) != 0 ||
      !offset.has_value() || *offset < 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> sub_inputs = sub_node.GetInputs();
  std::vector<Ort::ConstValueInfo> sub_outputs = sub_node.GetOutputs();
  if (sub_inputs.size() != 2 || sub_outputs.size() != 1 ||
      graph_output_names.count(Name(sub_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(sub_outputs[0], add_node) ||
      !IsIntTensorValueInfo(sub_outputs[0]) ||
      !IsIntTensorValueInfo(sub_inputs[0])) {
    return false;
  }
  Ort::ConstValueInfo source_input = sub_inputs[0];
  auto product_it = producers.find(Name(sub_inputs[1]));
  if (product_it == producers.end() || !IsOnnxOp(product_it->second, "Mul")) {
    return false;
  }
  Ort::ConstNode product_mul = product_it->second;
  if (accepted_node_ids.count(product_mul.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> product_inputs = product_mul.GetInputs();
  std::vector<Ort::ConstValueInfo> product_outputs = product_mul.GetOutputs();
  if (product_inputs.size() != 2 || product_outputs.size() != 1 ||
      graph_output_names.count(Name(product_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(product_outputs[0], sub_node) ||
      !IsIntTensorValueInfo(product_outputs[0])) {
    return false;
  }

  Ort::ConstNode div_node{nullptr};
  Ort::ConstValueInfo modulus_input{nullptr};
  for (Ort::ConstValueInfo input : product_inputs) {
    auto it = producers.find(Name(input));
    if (it != producers.end() && IsOnnxOp(it->second, "Div")) {
      div_node = it->second;
    } else {
      modulus_input = input;
    }
  }
  std::optional<int64_t> modulus = ReadScalarIntInitializer(modulus_input);
  if (!div_node || accepted_node_ids.count(div_node.GetId()) != 0 ||
      !modulus.has_value() || *modulus <= 0 ||
      *offset + *modulus > (*table_shape)[0]) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> div_inputs = div_node.GetInputs();
  std::vector<Ort::ConstValueInfo> div_outputs = div_node.GetOutputs();
  if (div_inputs.size() != 2 || div_outputs.size() != 1 ||
      graph_output_names.count(Name(div_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(div_outputs[0], product_mul) ||
      Name(div_inputs[0]) != Name(source_input) ||
      Name(div_inputs[1]) != Name(modulus_input) ||
      !IsIntTensorValueInfo(div_outputs[0])) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> mask_cast_inputs =
      mask_cast_node.GetInputs();
  std::vector<Ort::ConstValueInfo> mask_cast_outputs =
      mask_cast_node.GetOutputs();
  if (mask_cast_inputs.size() != 1 || mask_cast_outputs.size() != 1 ||
      graph_output_names.count(Name(mask_cast_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(mask_cast_outputs[0], final_mul) ||
      !IsIntTensorValueInfo(mask_cast_outputs[0])) {
    return false;
  }
  auto greater_it = producers.find(Name(mask_cast_inputs[0]));
  if (greater_it == producers.end() ||
      !IsOnnxOp(greater_it->second, "Greater")) {
    return false;
  }
  Ort::ConstNode greater_node = greater_it->second;
  if (accepted_node_ids.count(greater_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> greater_inputs = greater_node.GetInputs();
  std::vector<Ort::ConstValueInfo> greater_outputs = greater_node.GetOutputs();
  if (greater_inputs.size() != 2 || greater_outputs.size() != 1 ||
      graph_output_names.count(Name(greater_outputs[0])) != 0 ||
      !ValueHasOnlyConsumers(greater_outputs[0], mask_cast_node)) {
    return false;
  }

  Ort::ConstNode source_cast_node{nullptr};
  Ort::ConstValueInfo threshold_input{nullptr};
  for (Ort::ConstValueInfo input : greater_inputs) {
    auto it = producers.find(Name(input));
    if (it != producers.end() && IsOnnxOp(it->second, "Cast")) {
      source_cast_node = it->second;
    } else {
      threshold_input = input;
    }
  }
  if (!source_cast_node ||
      accepted_node_ids.count(source_cast_node.GetId()) != 0 ||
      !ReadScalarFloatInitializer(threshold_input).has_value() ||
      !CastTo(source_cast_node, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> source_cast_inputs =
      source_cast_node.GetInputs();
  std::vector<Ort::ConstValueInfo> source_cast_outputs =
      source_cast_node.GetOutputs();
  if (source_cast_inputs.size() != 1 || source_cast_outputs.size() != 1 ||
      graph_output_names.count(Name(source_cast_outputs[0])) != 0 ||
      !HasOnlyConsumer(source_cast_outputs[0], greater_node, 0) ||
      Name(source_cast_inputs[0]) != Name(source_input)) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  for (Ort::ConstNode node :
       {source_cast_node, greater_node, mask_cast_node, div_node, product_mul,
        sub_node, add_node, final_mul, gather_node, squeeze_node}) {
    if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }
  if (!FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                   selected_node_ids)) {
    return false;
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindBucketizeGatherFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);
  for (Ort::ConstNode squeeze_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseBucketizeGather(squeeze_node, producers, graph_output_names,
                                accepted_node_ids, fusion_nodes)) {
      continue;
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

}  // namespace musa_ep
