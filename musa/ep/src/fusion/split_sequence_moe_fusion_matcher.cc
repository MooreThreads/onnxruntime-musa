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
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

struct SliceMatch {
  Ort::ConstNode node{nullptr};
  Ort::ConstValueInfo data{nullptr};
  Ort::ConstValueInfo start{nullptr};
  Ort::ConstValueInfo end{nullptr};
};

struct GemmMatch {
  Ort::ConstNode node{nullptr};
  Ort::ConstValueInfo data{nullptr};
  Ort::ConstValueInfo weight{nullptr};
  Ort::ConstValueInfo bias{nullptr};
  int64_t input_width = 0;
  int64_t output_width = 0;
};

struct LayerNormBranchMatch {
  SliceMatch post_slice;
  Ort::ConstNode layer_norm{nullptr};
  float epsilon = 0.0f;
};

struct FfnBranchMatch {
  SliceMatch pre_slice;
  GemmMatch up;
  Ort::ConstNode relu{nullptr};
  GemmMatch down;
};

float ReadFloatAttribute(Ort::ConstNode node, const char* name,
                         float default_value) {
  Ort::ConstOpAttr attr;
  float value = default_value;
  return node.GetAttributeByName(name, attr).IsOK() &&
                 attr.GetValue(value).IsOK()
             ? value
             : default_value;
}

std::optional<std::vector<int64_t>> ReadIntConstant(
    Ort::ConstValueInfo value_info) {
  if (auto values = ReadIntInitializerNoLimit(value_info); values.has_value()) {
    return values;
  }

  Ort::ConstNode producer{nullptr};
  if (!GetProducer(value_info, producer) || !IsOnnxOp(producer, "Constant")) {
    return std::nullopt;
  }
  Ort::ConstOpAttr attr;
  Ort::Value value{nullptr};
  if (!producer.GetAttributeByName("value", attr).IsOK() ||
      !attr.GetTensorAttributeAsOrtValue(value).IsOK() || !value) {
    return std::nullopt;
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  const size_t count = static_cast<size_t>(info.GetElementCount());
  std::vector<int64_t> result;
  result.reserve(count);
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* data = value.GetTensorData<int64_t>();
    result.assign(data, data + count);
    return result;
  }
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t* data = value.GetTensorData<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      result.push_back(static_cast<int64_t>(data[i]));
    }
    return result;
  }
  return std::nullopt;
}

bool IsScalarIntegerTensor(Ort::ConstValueInfo value_info) {
  if (!IsIntTensorValueInfo(value_info)) {
    return false;
  }
  auto shape = GetStaticShape(value_info);
  return shape.has_value() &&
         (shape->empty() || (shape->size() == 1 && (*shape)[0] == 1));
}

bool IsBiasShape(const std::vector<int64_t>& shape, int64_t width) {
  return (shape.size() == 1 && shape[0] == width) ||
         (shape.size() == 2 && shape[0] == 1 && shape[1] == width);
}

bool SameFloat(float lhs, float rhs) {
  const float scale = std::max(std::fabs(lhs), std::fabs(rhs));
  return std::fabs(lhs - rhs) <= std::max(1.0e-12f, scale * 1.0e-6f);
}

bool MatchSlice(Ort::ConstNode slice, Ort::ConstValueInfo expected_data,
                Ort::ConstNode expected_consumer,
                int64_t expected_consumer_index,
                const std::unordered_set<std::string>& graph_output_names,
                const std::unordered_set<size_t>& accepted_node_ids,
                SliceMatch& match) {
  if (!IsOnnxOp(slice, "Slice") ||
      accepted_node_ids.count(slice.GetId()) != 0) {
    return false;
  }
  auto inputs = slice.GetInputs();
  auto outputs = slice.GetOutputs();
  if (inputs.size() != 5 || outputs.size() != 1 ||
      Name(inputs[0]) != Name(expected_data) ||
      !IsFloatTensorValueInfo(inputs[0]) ||
      !IsFloatTensorValueInfo(outputs[0]) ||
      !IsScalarIntegerTensor(inputs[1]) || !IsScalarIntegerTensor(inputs[2]) ||
      !HasSingleConsumerAt(outputs[0], expected_consumer,
                           expected_consumer_index, graph_output_names)) {
    return false;
  }

  auto axes = ReadIntConstant(inputs[3]);
  auto steps = ReadIntConstant(inputs[4]);
  auto input_shape = GetTensorShape(inputs[0]);
  auto output_shape = GetTensorShape(outputs[0]);
  if (!axes.has_value() || *axes != std::vector<int64_t>{0} ||
      !steps.has_value() || *steps != std::vector<int64_t>{1} ||
      !input_shape.has_value() || input_shape->size() != 2 ||
      !output_shape.has_value() || output_shape->size() != 2 ||
      !KnownDimsEqual(input_shape->back(), output_shape->back())) {
    return false;
  }

  match = {slice, inputs[0], inputs[1], inputs[2]};
  return true;
}

bool MatchGemm(Ort::ConstNode gemm, Ort::ConstNode expected_consumer,
               int64_t expected_consumer_index,
               const std::unordered_set<std::string>& graph_output_names,
               const std::unordered_set<size_t>& accepted_node_ids,
               GemmMatch& match) {
  if (!IsOnnxOp(gemm, "Gemm") || accepted_node_ids.count(gemm.GetId()) != 0 ||
      GetIntAttribute(gemm, "transA").value_or(0) != 0 ||
      GetIntAttribute(gemm, "transB").value_or(0) != 1 ||
      !SameFloat(ReadFloatAttribute(gemm, "alpha", 1.0f), 1.0f) ||
      !SameFloat(ReadFloatAttribute(gemm, "beta", 1.0f), 1.0f)) {
    return false;
  }

  auto inputs = gemm.GetInputs();
  auto outputs = gemm.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1 ||
      !IsFloatTensorValueInfo(inputs[0]) ||
      !IsFloatTensorValueInfo(inputs[1]) ||
      !IsFloatTensorValueInfo(inputs[2]) ||
      !IsFloatTensorValueInfo(outputs[0]) ||
      !inputs[1].IsConstantInitializer() ||
      !inputs[2].IsConstantInitializer() ||
      !HasSingleConsumerAt(outputs[0], expected_consumer,
                           expected_consumer_index, graph_output_names)) {
    return false;
  }

  auto data_shape = GetTensorShape(inputs[0]);
  auto output_shape = GetTensorShape(outputs[0]);
  auto weight_shape = GetStaticShape(inputs[1]);
  auto bias_shape = GetStaticShape(inputs[2]);
  if (!data_shape.has_value() || data_shape->size() != 2 ||
      !output_shape.has_value() || output_shape->size() != 2 ||
      !weight_shape.has_value() || weight_shape->size() != 2 ||
      !bias_shape.has_value()) {
    return false;
  }

  const int64_t input_width = (*weight_shape)[1];
  const int64_t output_width = (*weight_shape)[0];
  if (input_width <= 0 || output_width <= 0 ||
      !IsBiasShape(*bias_shape, output_width) ||
      !KnownDimsEqual(data_shape->back(), input_width) ||
      !KnownDimsEqual(output_shape->back(), output_width)) {
    return false;
  }

  match = {gemm, inputs[0], inputs[1], inputs[2], input_width, output_width};
  return true;
}

bool MatchLayerNormBranch(
    Ort::ConstNode final_concat, size_t input_index,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    LayerNormBranchMatch& match, Ort::ConstNode& residual_add) {
  auto concat_inputs = final_concat.GetInputs();
  if (input_index >= concat_inputs.size() ||
      !HasSingleConsumerAt(concat_inputs[input_index], final_concat,
                           static_cast<int64_t>(input_index),
                           graph_output_names)) {
    return false;
  }

  Ort::ConstNode layer_norm{nullptr};
  if (!GetProducer(concat_inputs[input_index], layer_norm) ||
      !IsOnnxOp(layer_norm, "LayerNormalization") ||
      accepted_node_ids.count(layer_norm.GetId()) != 0) {
    return false;
  }
  auto ln_inputs = layer_norm.GetInputs();
  auto ln_outputs = layer_norm.GetOutputs();
  if (ln_inputs.size() != 3 || ln_outputs.size() != 1 ||
      !IsFloatTensorValueInfo(ln_inputs[0]) ||
      !IsFloatTensorValueInfo(ln_inputs[1]) ||
      !IsFloatTensorValueInfo(ln_inputs[2]) ||
      !ln_inputs[1].IsConstantInitializer() ||
      !ln_inputs[2].IsConstantInitializer() ||
      GetIntAttribute(layer_norm, "axis").value_or(-1) != -1) {
    return false;
  }

  auto scale_shape = GetStaticShape(ln_inputs[1]);
  auto bias_shape = GetStaticShape(ln_inputs[2]);
  auto output_shape = GetTensorShape(ln_outputs[0]);
  const float epsilon = ReadFloatAttribute(layer_norm, "epsilon", 1.0e-5f);
  if (!scale_shape.has_value() || scale_shape->size() != 1 ||
      (*scale_shape)[0] <= 0 || !bias_shape.has_value() ||
      *bias_shape != *scale_shape || !output_shape.has_value() ||
      output_shape->size() != 2 ||
      !KnownDimsEqual(output_shape->back(), (*scale_shape)[0]) ||
      !std::isfinite(epsilon) || epsilon <= 0.0f) {
    return false;
  }

  Ort::ConstNode post_slice{nullptr};
  if (!GetProducer(ln_inputs[0], post_slice)) {
    return false;
  }
  auto slice_inputs = post_slice.GetInputs();
  if (slice_inputs.empty()) {
    return false;
  }

  Ort::ConstNode branch_residual{nullptr};
  if (!GetProducer(slice_inputs[0], branch_residual) ||
      !IsOnnxOp(branch_residual, "Add") ||
      accepted_node_ids.count(branch_residual.GetId()) != 0) {
    return false;
  }
  if (residual_add && residual_add.GetId() != branch_residual.GetId()) {
    return false;
  }

  SliceMatch slice_match;
  if (!MatchSlice(post_slice, slice_inputs[0], layer_norm, 0,
                  graph_output_names, accepted_node_ids, slice_match)) {
    return false;
  }
  residual_add = branch_residual;
  match = {slice_match, layer_norm, epsilon};
  return true;
}

bool MatchFfnBranch(Ort::ConstNode ffn_concat, size_t input_index,
                    Ort::ConstValueInfo x,
                    const LayerNormBranchMatch& layer_norm_branch,
                    const std::unordered_set<std::string>& graph_output_names,
                    const std::unordered_set<size_t>& accepted_node_ids,
                    FfnBranchMatch& match) {
  auto concat_inputs = ffn_concat.GetInputs();
  if (input_index >= concat_inputs.size() ||
      !HasSingleConsumerAt(concat_inputs[input_index], ffn_concat,
                           static_cast<int64_t>(input_index),
                           graph_output_names)) {
    return false;
  }

  Ort::ConstNode down{nullptr};
  if (!GetProducer(concat_inputs[input_index], down)) {
    return false;
  }
  GemmMatch down_match;
  if (!MatchGemm(down, ffn_concat, static_cast<int64_t>(input_index),
                 graph_output_names, accepted_node_ids, down_match)) {
    return false;
  }

  Ort::ConstNode relu{nullptr};
  if (!GetProducer(down_match.data, relu) || !IsOnnxOp(relu, "Relu") ||
      accepted_node_ids.count(relu.GetId()) != 0) {
    return false;
  }
  auto relu_inputs = relu.GetInputs();
  auto relu_outputs = relu.GetOutputs();
  if (relu_inputs.size() != 1 || relu_outputs.size() != 1 ||
      Name(relu_outputs[0]) != Name(down_match.data) ||
      !HasSingleConsumerAt(relu_outputs[0], down, 0, graph_output_names)) {
    return false;
  }

  Ort::ConstNode up{nullptr};
  if (!GetProducer(relu_inputs[0], up)) {
    return false;
  }
  GemmMatch up_match;
  if (!MatchGemm(up, relu, 0, graph_output_names, accepted_node_ids,
                 up_match)) {
    return false;
  }

  Ort::ConstNode pre_slice{nullptr};
  if (!GetProducer(up_match.data, pre_slice)) {
    return false;
  }
  SliceMatch pre_slice_match;
  if (!MatchSlice(pre_slice, x, up, 0, graph_output_names, accepted_node_ids,
                  pre_slice_match) ||
      Name(pre_slice_match.start) != Name(layer_norm_branch.post_slice.start) ||
      Name(pre_slice_match.end) != Name(layer_norm_branch.post_slice.end) ||
      up_match.input_width != down_match.output_width ||
      up_match.output_width != down_match.input_width) {
    return false;
  }

  match = {pre_slice_match, up_match, relu, down_match};
  return true;
}

bool CanFuseSplitSequenceMoE(
    Ort::ConstNode final_concat,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(final_concat, "Concat") ||
      accepted_node_ids.count(final_concat.GetId()) != 0) {
    return false;
  }
  auto final_inputs = final_concat.GetInputs();
  auto final_outputs = final_concat.GetOutputs();
  auto final_shape =
      final_outputs.empty() ? std::nullopt : GetTensorShape(final_outputs[0]);
  int64_t final_axis = 0;
  if (final_inputs.size() < 2 || final_outputs.size() != 1 ||
      !IsFloatTensorValueInfo(final_outputs[0]) || !final_shape.has_value() ||
      final_shape->size() != 2 ||
      !NormalizeAxis(GetIntAttribute(final_concat, "axis").value_or(0), 2,
                     final_axis) ||
      final_axis != 0) {
    return false;
  }

  std::vector<LayerNormBranchMatch> layer_norm_branches;
  layer_norm_branches.reserve(final_inputs.size());
  Ort::ConstNode residual_add{nullptr};
  for (size_t i = 0; i < final_inputs.size(); ++i) {
    LayerNormBranchMatch branch;
    if (!MatchLayerNormBranch(final_concat, i, graph_output_names,
                              accepted_node_ids, branch, residual_add)) {
      return false;
    }
    if (!layer_norm_branches.empty() &&
        !SameFloat(branch.epsilon, layer_norm_branches.front().epsilon)) {
      return false;
    }
    layer_norm_branches.push_back(std::move(branch));
  }

  auto residual_inputs = residual_add.GetInputs();
  auto residual_outputs = residual_add.GetOutputs();
  if (residual_inputs.size() != 2 || residual_outputs.size() != 1 ||
      !IsFloatTensorValueInfo(residual_outputs[0]) ||
      graph_output_names.count(Name(residual_outputs[0])) != 0 ||
      residual_outputs[0].GetConsumers().size() != final_inputs.size()) {
    return false;
  }

  Ort::ConstNode ffn_concat{nullptr};
  Ort::ConstValueInfo x{nullptr};
  int64_t ffn_input_index = -1;
  for (int64_t i = 0; i < 2; ++i) {
    Ort::ConstNode producer{nullptr};
    if (GetProducer(residual_inputs[static_cast<size_t>(i)], producer) &&
        IsOnnxOp(producer, "Concat") &&
        producer.GetInputs().size() == final_inputs.size()) {
      if (ffn_concat) {
        return false;
      }
      ffn_concat = producer;
      ffn_input_index = i;
    }
  }
  if (!ffn_concat || accepted_node_ids.count(ffn_concat.GetId()) != 0) {
    return false;
  }
  x = residual_inputs[static_cast<size_t>(1 - ffn_input_index)];
  auto x_shape = GetTensorShape(x);
  auto ffn_outputs = ffn_concat.GetOutputs();
  int64_t ffn_axis = 0;
  if (!IsFloatTensorValueInfo(x) || !x_shape.has_value() ||
      x_shape->size() != 2 || x_shape->back() <= 0 || ffn_outputs.size() != 1 ||
      !HasSingleConsumerAt(ffn_outputs[0], residual_add, ffn_input_index,
                           graph_output_names) ||
      !NormalizeAxis(GetIntAttribute(ffn_concat, "axis").value_or(0), 2,
                     ffn_axis) ||
      ffn_axis != 0) {
    return false;
  }

  std::vector<FfnBranchMatch> ffn_branches;
  ffn_branches.reserve(final_inputs.size());
  for (size_t i = 0; i < final_inputs.size(); ++i) {
    FfnBranchMatch branch;
    if (!MatchFfnBranch(ffn_concat, i, x, layer_norm_branches[i],
                        graph_output_names, accepted_node_ids, branch)) {
      return false;
    }
    const int64_t width = x_shape->back();
    if (branch.up.input_width != width || branch.down.output_width != width) {
      return false;
    }
    if (!ffn_branches.empty() &&
        (branch.up.input_width != ffn_branches.front().up.input_width ||
         branch.up.output_width != ffn_branches.front().up.output_width ||
         branch.down.input_width != ffn_branches.front().down.input_width ||
         branch.down.output_width != ffn_branches.front().down.output_width)) {
      return false;
    }
    if (i > 0 && Name(branch.pre_slice.start) !=
                     Name(ffn_branches.back().pre_slice.end)) {
      return false;
    }
    ffn_branches.push_back(std::move(branch));
  }

  std::unordered_set<size_t> post_slice_ids;
  for (const LayerNormBranchMatch& branch : layer_norm_branches) {
    post_slice_ids.insert(branch.post_slice.node.GetId());
  }
  for (const auto& consumer : residual_outputs[0].GetConsumers()) {
    if (post_slice_ids.count(consumer.node.GetId()) == 0 ||
        consumer.index != 0) {
      return false;
    }
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  fusion_nodes.reserve(final_inputs.size() * 6 + 3);
  for (const FfnBranchMatch& branch : ffn_branches) {
    for (Ort::ConstNode node : {branch.pre_slice.node, branch.up.node,
                                branch.relu, branch.down.node}) {
      if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                         fusion_nodes)) {
        return false;
      }
    }
  }
  for (Ort::ConstNode node : {ffn_concat, residual_add}) {
    if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }
  for (const LayerNormBranchMatch& branch : layer_norm_branches) {
    for (Ort::ConstNode node : {branch.post_slice.node, branch.layer_norm}) {
      if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                         fusion_nodes)) {
        return false;
      }
    }
  }
  if (!AddFusionNode(final_concat, accepted_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }
  return FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                     selected_node_ids);
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindSplitSequenceMoEFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode node : all_nodes) {
    if (!IsOnnxOp(node, "Concat")) {
      continue;
    }
    std::vector<Ort::ConstNode> fusion_nodes;
    if (CanFuseSplitSequenceMoE(node, graph_output_names, accepted_node_ids,
                                fusion_nodes)) {
      fusions.push_back(std::move(fusion_nodes));
    }
  }
  return fusions;
}

}  // namespace musa_ep
