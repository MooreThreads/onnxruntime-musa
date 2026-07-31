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

enum class MoELinearKind { Gemm, MatMulAdd };

struct MoELinearMatch {
  MoELinearKind kind = MoELinearKind::Gemm;
  Ort::ConstNode linear{nullptr};
  Ort::ConstNode add{nullptr};
  Ort::ConstValueInfo input{nullptr};
  std::vector<int64_t> weight_shape;
  int64_t input_width = 0;
  int64_t output_width = 0;
};

struct MoEBranchMatch {
  MoELinearMatch first_linear;
  Ort::ConstNode first_relu{nullptr};
  MoELinearMatch second_linear;
  Ort::ConstNode second_relu{nullptr};
  Ort::ConstNode unsqueeze{nullptr};
  std::string x_name;
  std::vector<int64_t> x_shape;
};

struct MoERouterMatch {
  Ort::ConstNode mul{nullptr};
  Ort::ConstNode reduce_sum{nullptr};
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

bool IsSupportedGemm(Ort::ConstNode gemm) {
  return IsOnnxOp(gemm, "Gemm") &&
         GetIntAttribute(gemm, "transA").value_or(0) == 0 &&
         GetIntAttribute(gemm, "transB").value_or(0) == 1 &&
         ReadFloatAttribute(gemm, "alpha", 1.0f) == 1.0f &&
         ReadFloatAttribute(gemm, "beta", 1.0f) == 1.0f;
}

bool IsBiasShape(const std::vector<int64_t>& shape, int64_t width) {
  return (shape.size() == 1 && shape[0] == width) ||
         (shape.size() == 2 && shape[0] == 1 && shape[1] == width);
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

std::optional<std::vector<int64_t>> ReadAxes(Ort::ConstNode node) {
  auto inputs = node.GetInputs();
  if (inputs.size() >= 2) {
    return ReadIntConstant(inputs[1]);
  }
  return GetIntsAttribute(node, "axes");
}

bool ShapeIsPrefixAndWidth(const std::vector<int64_t>& shape,
                           const std::vector<int64_t>& prefix_source,
                           int64_t width) {
  if (shape.size() != prefix_source.size() || shape.empty() ||
      !KnownDimsEqual(shape.back(), width)) {
    return false;
  }
  for (size_t i = 0; i + 1 < shape.size(); ++i) {
    if (!KnownDimsEqual(shape[i], prefix_source[i])) {
      return false;
    }
  }
  return true;
}

bool MatchLinear(Ort::ConstValueInfo output, Ort::ConstNode expected_consumer,
                 int64_t expected_consumer_index,
                 const std::unordered_set<std::string>& graph_output_names,
                 const std::unordered_set<size_t>& accepted_node_ids,
                 MoELinearMatch& match) {
  if (!IsFloatTensorValueInfo(output)) {
    return false;
  }
  Ort::ConstNode producer{nullptr};
  if (!GetProducer(output, producer) ||
      accepted_node_ids.count(producer.GetId()) != 0) {
    return false;
  }

  Ort::ConstNode linear = producer;
  Ort::ConstNode add{nullptr};
  Ort::ConstValueInfo data_input{nullptr};
  Ort::ConstValueInfo weight{nullptr};
  Ort::ConstValueInfo bias{nullptr};
  MoELinearKind kind = MoELinearKind::Gemm;

  if (IsSupportedGemm(linear)) {
    auto inputs = linear.GetInputs();
    auto outputs = linear.GetOutputs();
    if (inputs.size() != 3 || outputs.size() != 1 ||
        Name(outputs[0]) != Name(output) ||
        !HasSingleConsumerAt(outputs[0], expected_consumer,
                             expected_consumer_index, graph_output_names)) {
      return false;
    }
    data_input = inputs[0];
    weight = inputs[1];
    bias = inputs[2];
  } else if (IsOnnxOp(producer, "Add")) {
    add = producer;
    auto add_inputs = add.GetInputs();
    auto add_outputs = add.GetOutputs();
    if (add_inputs.size() != 2 || add_outputs.size() != 1 ||
        Name(add_outputs[0]) != Name(output) ||
        !HasSingleConsumerAt(add_outputs[0], expected_consumer,
                             expected_consumer_index, graph_output_names)) {
      return false;
    }

    int64_t matmul_input_index = -1;
    for (int64_t i = 0; i < 2; ++i) {
      Ort::ConstNode input_producer{nullptr};
      if (GetProducer(add_inputs[static_cast<size_t>(i)], input_producer) &&
          IsOnnxOp(input_producer, "MatMul")) {
        if (matmul_input_index >= 0) {
          return false;
        }
        matmul_input_index = i;
        linear = input_producer;
      }
    }
    if (matmul_input_index < 0 ||
        accepted_node_ids.count(linear.GetId()) != 0) {
      return false;
    }
    const size_t bias_input_index = static_cast<size_t>(1 - matmul_input_index);
    auto matmul_inputs = linear.GetInputs();
    auto matmul_outputs = linear.GetOutputs();
    if (matmul_inputs.size() != 2 || matmul_outputs.size() != 1 ||
        Name(matmul_outputs[0]) !=
            Name(add_inputs[static_cast<size_t>(matmul_input_index)]) ||
        !HasSingleConsumerAt(matmul_outputs[0], add, matmul_input_index,
                             graph_output_names)) {
      return false;
    }
    data_input = matmul_inputs[0];
    weight = matmul_inputs[1];
    bias = add_inputs[bias_input_index];
    kind = MoELinearKind::MatMulAdd;
  } else {
    return false;
  }

  if (!IsFloatTensorValueInfo(data_input) || !IsFloatTensorValueInfo(weight) ||
      !IsFloatTensorValueInfo(bias) || !weight.IsConstantInitializer() ||
      !bias.IsConstantInitializer()) {
    return false;
  }
  auto weight_shape = GetStaticShape(weight);
  auto bias_shape = GetStaticShape(bias);
  if (!weight_shape.has_value() || weight_shape->size() != 2 ||
      !bias_shape.has_value()) {
    return false;
  }

  const int64_t input_width =
      kind == MoELinearKind::Gemm ? (*weight_shape)[1] : (*weight_shape)[0];
  const int64_t output_width =
      kind == MoELinearKind::Gemm ? (*weight_shape)[0] : (*weight_shape)[1];
  if (input_width <= 0 || output_width <= 0 ||
      !IsBiasShape(*bias_shape, output_width)) {
    return false;
  }

  match = {kind,        linear,      add, data_input, std::move(*weight_shape),
           input_width, output_width};
  return true;
}

bool MatchBranch(Ort::ConstNode concat, size_t concat_input_index,
                 Ort::ConstValueInfo concat_input, int64_t concat_axis,
                 const std::unordered_set<std::string>& graph_output_names,
                 const std::unordered_set<size_t>& accepted_node_ids,
                 MoEBranchMatch& branch) {
  if (!IsFloatTensorValueInfo(concat_input) ||
      !HasSingleConsumerAt(concat_input, concat,
                           static_cast<int64_t>(concat_input_index),
                           graph_output_names)) {
    return false;
  }

  Ort::ConstNode unsqueeze{nullptr};
  if (!GetProducer(concat_input, unsqueeze) ||
      !IsOnnxOp(unsqueeze, "Unsqueeze") ||
      accepted_node_ids.count(unsqueeze.GetId()) != 0) {
    return false;
  }
  auto unsqueeze_inputs = unsqueeze.GetInputs();
  auto unsqueeze_outputs = unsqueeze.GetOutputs();
  auto unsqueeze_axes = ReadAxes(unsqueeze);
  auto unsqueeze_input_shape = unsqueeze_inputs.empty()
                                   ? std::nullopt
                                   : GetTensorShape(unsqueeze_inputs[0]);
  auto unsqueeze_output_shape = unsqueeze_outputs.empty()
                                    ? std::nullopt
                                    : GetTensorShape(unsqueeze_outputs[0]);
  int64_t unsqueeze_axis = 0;
  if ((unsqueeze_inputs.size() != 1 && unsqueeze_inputs.size() != 2) ||
      unsqueeze_outputs.size() != 1 || !unsqueeze_axes.has_value() ||
      unsqueeze_axes->size() != 1 || !unsqueeze_input_shape.has_value() ||
      !unsqueeze_output_shape.has_value() ||
      unsqueeze_input_shape->size() < 2 ||
      unsqueeze_output_shape->size() != unsqueeze_input_shape->size() + 1 ||
      !NormalizeAxis((*unsqueeze_axes)[0], unsqueeze_output_shape->size(),
                     unsqueeze_axis) ||
      unsqueeze_axis != concat_axis ||
      (*unsqueeze_output_shape)[static_cast<size_t>(unsqueeze_axis)] != 1) {
    return false;
  }

  Ort::ConstNode second_relu{nullptr};
  if (!GetProducer(unsqueeze_inputs[0], second_relu) ||
      !IsOnnxOp(second_relu, "Relu") ||
      accepted_node_ids.count(second_relu.GetId()) != 0) {
    return false;
  }
  auto second_relu_inputs = second_relu.GetInputs();
  auto second_relu_outputs = second_relu.GetOutputs();
  if (second_relu_inputs.size() != 1 || second_relu_outputs.size() != 1 ||
      Name(second_relu_outputs[0]) != Name(unsqueeze_inputs[0]) ||
      !HasSingleConsumerAt(second_relu_outputs[0], unsqueeze, 0,
                           graph_output_names)) {
    return false;
  }

  MoELinearMatch second_linear;
  if (!MatchLinear(second_relu_inputs[0], second_relu, 0, graph_output_names,
                   accepted_node_ids, second_linear)) {
    return false;
  }

  Ort::ConstNode first_relu{nullptr};
  if (!GetProducer(second_linear.input, first_relu) ||
      !IsOnnxOp(first_relu, "Relu") ||
      accepted_node_ids.count(first_relu.GetId()) != 0) {
    return false;
  }
  auto first_relu_inputs = first_relu.GetInputs();
  auto first_relu_outputs = first_relu.GetOutputs();
  if (first_relu_inputs.size() != 1 || first_relu_outputs.size() != 1 ||
      Name(first_relu_outputs[0]) != Name(second_linear.input) ||
      !HasSingleConsumerAt(first_relu_outputs[0], second_linear.linear, 0,
                           graph_output_names)) {
    return false;
  }

  MoELinearMatch first_linear;
  if (!MatchLinear(first_relu_inputs[0], first_relu, 0, graph_output_names,
                   accepted_node_ids, first_linear)) {
    return false;
  }

  auto x_shape = GetTensorShape(first_linear.input);
  auto first_relu_shape = GetTensorShape(first_relu_outputs[0]);
  if (!x_shape.has_value() || x_shape->size() < 2 ||
      !first_relu_shape.has_value() ||
      !KnownDimsEqual(x_shape->back(), first_linear.input_width) ||
      first_linear.output_width != second_linear.input_width ||
      !ShapeIsPrefixAndWidth(*first_relu_shape, *x_shape,
                             first_linear.output_width) ||
      !ShapeIsPrefixAndWidth(*unsqueeze_input_shape, *x_shape,
                             second_linear.output_width)) {
    return false;
  }

  std::vector<int64_t> expected_unsqueeze_shape = *unsqueeze_input_shape;
  expected_unsqueeze_shape.insert(
      expected_unsqueeze_shape.begin() + unsqueeze_axis, 1);
  if (!ShapesEqualOnKnownDims(*unsqueeze_output_shape,
                              expected_unsqueeze_shape)) {
    return false;
  }

  const std::string x_name = Name(first_linear.input);
  branch = {std::move(first_linear),
            first_relu,
            std::move(second_linear),
            second_relu,
            unsqueeze,
            x_name,
            std::move(*x_shape)};
  return true;
}

bool MatchRouterTail(Ort::ConstValueInfo concat_output,
                     const Ort::ValueInfoConsumerProducerInfo& concat_consumer,
                     int64_t concat_axis,
                     const std::vector<int64_t>& concat_shape,
                     const std::vector<int64_t>& router_shape,
                     const std::vector<int64_t>& output_shape,
                     const std::unordered_set<std::string>& graph_output_names,
                     const std::unordered_set<size_t>& accepted_node_ids,
                     MoERouterMatch& router_match) {
  Ort::ConstNode mul = concat_consumer.node;
  if (!IsOnnxOp(mul, "Mul") || concat_consumer.index < 0 ||
      concat_consumer.index > 1 || accepted_node_ids.count(mul.GetId()) != 0) {
    return false;
  }
  auto mul_inputs = mul.GetInputs();
  auto mul_outputs = mul.GetOutputs();
  const size_t concat_input_index = static_cast<size_t>(concat_consumer.index);
  if (mul_inputs.size() != 2 || mul_outputs.size() != 1 ||
      Name(mul_inputs[concat_input_index]) != Name(concat_output) ||
      !IsFloatTensorValueInfo(mul_outputs[0])) {
    return false;
  }
  auto actual_mul_shape = GetTensorShape(mul_outputs[0]);
  auto actual_router_shape = GetTensorShape(mul_inputs[1 - concat_input_index]);
  if (!actual_mul_shape.has_value() || !actual_router_shape.has_value() ||
      !ShapesEqualOnKnownDims(*actual_mul_shape, concat_shape) ||
      !ShapesEqualOnKnownDims(*actual_router_shape, router_shape) ||
      !IsFloatTensorValueInfo(mul_inputs[1 - concat_input_index])) {
    return false;
  }

  auto mul_consumers = mul_outputs[0].GetConsumers();
  if (mul_consumers.size() != 1 || mul_consumers[0].index != 0 ||
      graph_output_names.count(Name(mul_outputs[0])) != 0) {
    return false;
  }
  Ort::ConstNode reduce_sum = mul_consumers[0].node;
  if (!IsOnnxOp(reduce_sum, "ReduceSum") ||
      accepted_node_ids.count(reduce_sum.GetId()) != 0 ||
      GetIntAttribute(reduce_sum, "keepdims").value_or(1) != 0) {
    return false;
  }
  auto reduce_inputs = reduce_sum.GetInputs();
  auto reduce_outputs = reduce_sum.GetOutputs();
  auto reduce_axes = ReadAxes(reduce_sum);
  int64_t reduce_axis = 0;
  if ((reduce_inputs.size() != 1 && reduce_inputs.size() != 2) ||
      reduce_outputs.size() != 1 ||
      Name(reduce_inputs[0]) != Name(mul_outputs[0]) ||
      !reduce_axes.has_value() || reduce_axes->size() != 1 ||
      !NormalizeAxis((*reduce_axes)[0], concat_shape.size(), reduce_axis) ||
      reduce_axis != concat_axis ||
      !IsFloatTensorValueInfo(reduce_outputs[0])) {
    return false;
  }
  auto actual_output_shape = GetTensorShape(reduce_outputs[0]);
  if (!actual_output_shape.has_value() ||
      !ShapesEqualOnKnownDims(*actual_output_shape, output_shape)) {
    return false;
  }

  router_match = {mul, reduce_sum};
  return true;
}

bool CanFuseMoE(Ort::ConstNode candidate_reduce_sum,
                const std::unordered_set<std::string>& graph_output_names,
                const std::unordered_set<size_t>& accepted_node_ids,
                std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(candidate_reduce_sum, "ReduceSum") ||
      accepted_node_ids.count(candidate_reduce_sum.GetId()) != 0) {
    return false;
  }
  auto candidate_inputs = candidate_reduce_sum.GetInputs();
  Ort::ConstNode candidate_mul{nullptr};
  if (candidate_inputs.empty() ||
      !GetProducer(candidate_inputs[0], candidate_mul) ||
      !IsOnnxOp(candidate_mul, "Mul")) {
    return false;
  }

  Ort::ConstNode concat{nullptr};
  for (Ort::ConstValueInfo input : candidate_mul.GetInputs()) {
    Ort::ConstNode producer{nullptr};
    if (GetProducer(input, producer) && IsOnnxOp(producer, "Concat")) {
      if (concat) {
        return false;
      }
      concat = producer;
    }
  }
  if (!concat || accepted_node_ids.count(concat.GetId()) != 0) {
    return false;
  }

  auto concat_inputs = concat.GetInputs();
  auto concat_outputs = concat.GetOutputs();
  auto concat_axis_attr = GetIntAttribute(concat, "axis");
  auto concat_shape =
      concat_outputs.empty() ? std::nullopt : GetTensorShape(concat_outputs[0]);
  int64_t concat_axis = 0;
  if (concat_inputs.size() < 2 || concat_outputs.size() != 1 ||
      !concat_axis_attr.has_value() || !concat_shape.has_value() ||
      concat_shape->size() < 3 ||
      !NormalizeAxis(*concat_axis_attr, concat_shape->size(), concat_axis) ||
      (concat_axis != static_cast<int64_t>(concat_shape->size()) - 2 &&
       concat_axis != static_cast<int64_t>(concat_shape->size()) - 1) ||
      graph_output_names.count(Name(concat_outputs[0])) != 0) {
    return false;
  }

  std::vector<MoEBranchMatch> branches;
  branches.reserve(concat_inputs.size());
  for (size_t i = 0; i < concat_inputs.size(); ++i) {
    MoEBranchMatch branch;
    if (!MatchBranch(concat, i, concat_inputs[i], concat_axis,
                     graph_output_names, accepted_node_ids, branch)) {
      return false;
    }
    if (!branches.empty()) {
      const MoEBranchMatch& first = branches.front();
      if (branch.x_name != first.x_name || branch.x_shape != first.x_shape ||
          branch.first_linear.input_width != first.first_linear.input_width ||
          branch.first_linear.output_width != first.first_linear.output_width ||
          branch.second_linear.input_width != first.second_linear.input_width ||
          branch.second_linear.output_width !=
              first.second_linear.output_width) {
        return false;
      }
    }
    branches.push_back(std::move(branch));
  }

  const int64_t expert_count = static_cast<int64_t>(branches.size());
  const MoEBranchMatch& first_branch = branches.front();
  const size_t branch_rank = first_branch.x_shape.size();
  const int64_t output_width = first_branch.second_linear.output_width;
  std::vector<int64_t> expected_concat_shape = first_branch.x_shape;
  expected_concat_shape.back() = output_width;
  expected_concat_shape.insert(expected_concat_shape.begin() + concat_axis,
                               expert_count);
  if (!ShapesEqualOnKnownDims(*concat_shape, expected_concat_shape)) {
    return false;
  }

  const size_t output_axis = concat_axis == static_cast<int64_t>(branch_rank)
                                 ? branch_rank - 1
                                 : branch_rank;
  std::vector<int64_t> expected_router_shape = expected_concat_shape;
  expected_router_shape[output_axis] = 1;
  std::vector<int64_t> expected_output_shape = first_branch.x_shape;
  expected_output_shape.back() = output_width;

  auto concat_consumers = concat_outputs[0].GetConsumers();
  if (concat_consumers.empty()) {
    return false;
  }
  std::vector<MoERouterMatch> routers;
  routers.reserve(concat_consumers.size());
  std::unordered_set<size_t> router_node_ids;
  bool includes_candidate = false;
  for (const auto& consumer : concat_consumers) {
    MoERouterMatch router;
    if (!MatchRouterTail(concat_outputs[0], consumer, concat_axis,
                         expected_concat_shape, expected_router_shape,
                         expected_output_shape, graph_output_names,
                         accepted_node_ids, router) ||
        !router_node_ids.insert(router.mul.GetId()).second) {
      return false;
    }
    includes_candidate |=
        router.reduce_sum.GetId() == candidate_reduce_sum.GetId();
    routers.push_back(router);
  }
  if (!includes_candidate) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  fusion_nodes.reserve(branches.size() * 7 + 1 + routers.size() * 2);
  for (const MoEBranchMatch& branch : branches) {
    std::vector<Ort::ConstNode> nodes = {branch.first_linear.linear};
    if (branch.first_linear.add) {
      nodes.push_back(branch.first_linear.add);
    }
    nodes.push_back(branch.first_relu);
    nodes.push_back(branch.second_linear.linear);
    if (branch.second_linear.add) {
      nodes.push_back(branch.second_linear.add);
    }
    nodes.push_back(branch.second_relu);
    nodes.push_back(branch.unsqueeze);
    for (Ort::ConstNode node : nodes) {
      if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                         fusion_nodes)) {
        return false;
      }
    }
  }
  if (!AddFusionNode(concat, accepted_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }
  for (const MoERouterMatch& router : routers) {
    for (Ort::ConstNode node : {router.mul, router.reduce_sum}) {
      if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                         fusion_nodes)) {
        return false;
      }
    }
  }
  return FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                     selected_node_ids);
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindMoEFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  std::unordered_set<size_t> emitted_concat_ids;
  for (Ort::ConstNode node : all_nodes) {
    if (!IsOnnxOp(node, "ReduceSum")) {
      continue;
    }
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseMoE(node, graph_output_names, accepted_node_ids,
                    fusion_nodes)) {
      continue;
    }
    Ort::ConstNode concat{nullptr};
    for (Ort::ConstNode fusion_node : fusion_nodes) {
      if (IsOnnxOp(fusion_node, "Concat")) {
        concat = fusion_node;
        break;
      }
    }
    if (concat && emitted_concat_ids.insert(concat.GetId()).second) {
      fusions.push_back(std::move(fusion_nodes));
    }
  }
  return fusions;
}

}  // namespace musa_ep
