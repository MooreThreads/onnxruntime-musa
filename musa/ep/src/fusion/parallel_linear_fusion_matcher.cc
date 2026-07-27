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

#include <map>
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

struct ParallelLinearBranch {
  Ort::ConstNode linear{nullptr};
  Ort::ConstNode add{nullptr};
  Ort::ConstNode activation{nullptr};
};

int64_t ReadIntAttribute(Ort::ConstNode node, const char* name,
                         int64_t default_value) {
  Ort::ConstOpAttr attr;
  if (!node.GetAttributeByName(name, attr).IsOK()) {
    return default_value;
  }
  int64_t value = default_value;
  return attr.GetValue(value).IsOK() ? value : default_value;
}

float ReadFloatAttribute(Ort::ConstNode node, const char* name,
                         float default_value) {
  Ort::ConstOpAttr attr;
  if (!node.GetAttributeByName(name, attr).IsOK()) {
    return default_value;
  }
  float value = default_value;
  return attr.GetValue(value).IsOK() ? value : default_value;
}

bool ParseBranch(Ort::ConstNode matmul,
                 const std::unordered_set<std::string>& graph_output_names,
                 const std::unordered_set<size_t>& accepted_node_ids,
                 ParallelLinearBranch& branch, std::string& group_key) {
  const bool is_matmul = IsOnnxOp(matmul, "MatMul");
  const bool is_gemm = IsOnnxOp(matmul, "Gemm");
  if ((!is_matmul && !is_gemm) ||
      accepted_node_ids.count(matmul.GetId()) != 0) {
    return false;
  }
  auto matmul_inputs = matmul.GetInputs();
  auto matmul_outputs = matmul.GetOutputs();
  if ((is_matmul && matmul_inputs.size() != 2) ||
      (is_gemm && matmul_inputs.size() != 2 && matmul_inputs.size() != 3) ||
      matmul_outputs.size() != 1 || !IsFloatTensorValueInfo(matmul_inputs[0]) ||
      !IsFloatTensorValueInfo(matmul_inputs[1]) ||
      !matmul_inputs[1].IsConstantInitializer()) {
    return false;
  }
  if (is_gemm && (ReadIntAttribute(matmul, "transA", 0) != 0 ||
                  ReadIntAttribute(matmul, "transB", 0) != 0 ||
                  ReadFloatAttribute(matmul, "alpha", 1.0f) != 1.0f ||
                  ReadFloatAttribute(matmul, "beta", 1.0f) != 1.0f)) {
    return false;
  }
  auto weight_shape = GetStaticShape(matmul_inputs[1]);
  if (!weight_shape.has_value() || weight_shape->size() != 2 ||
      (*weight_shape)[0] <= 0 || (*weight_shape)[1] <= 0) {
    return false;
  }

  Ort::ConstNode add{nullptr};
  Ort::ConstValueInfo bias{nullptr};
  Ort::ConstValueInfo linear_output = matmul_outputs[0];
  if (is_matmul) {
    auto matmul_consumers = matmul_outputs[0].GetConsumers();
    if (matmul_consumers.size() == 1 && matmul_consumers[0].index >= 0 &&
        matmul_consumers[0].index <= 1 &&
        IsOnnxOp(matmul_consumers[0].node, "Add")) {
      Ort::ConstNode candidate_add = matmul_consumers[0].node;
      auto add_inputs = candidate_add.GetInputs();
      auto add_outputs = candidate_add.GetOutputs();
      const size_t bias_index =
          static_cast<size_t>(1 - matmul_consumers[0].index);
      const auto candidate_bias = add_inputs.size() == 2
                                      ? add_inputs[bias_index]
                                      : Ort::ConstValueInfo{nullptr};
      const auto candidate_bias_shape = candidate_bias != nullptr
                                            ? GetStaticShape(candidate_bias)
                                            : std::nullopt;
      const bool candidate_bias_valid =
          add_inputs.size() == 2 && add_outputs.size() == 1 &&
          accepted_node_ids.count(candidate_add.GetId()) == 0 &&
          IsFloatTensorValueInfo(candidate_bias) &&
          candidate_bias.IsConstantInitializer() &&
          candidate_bias_shape.has_value() &&
          ((candidate_bias_shape->size() == 1 &&
            (*candidate_bias_shape)[0] == weight_shape->back()) ||
           (candidate_bias_shape->size() == 2 &&
            (*candidate_bias_shape)[0] == 1 &&
            (*candidate_bias_shape)[1] == weight_shape->back()));
      if (candidate_bias_valid) {
        add = candidate_add;
        bias = candidate_bias;
        linear_output = add_outputs[0];
      }
    }
  } else if (matmul_inputs.size() == 3) {
    bias = matmul_inputs[2];
  }
  const bool has_bias = bias != nullptr;
  if (has_bias &&
      (!IsFloatTensorValueInfo(bias) || !bias.IsConstantInitializer())) {
    return false;
  }
  const auto bias_shape = has_bias ? GetStaticShape(bias) : std::nullopt;
  const int64_t n = weight_shape->back();
  if (has_bias && (!bias_shape.has_value() ||
                   !((bias_shape->size() == 1 && (*bias_shape)[0] == n) ||
                     (bias_shape->size() == 2 && (*bias_shape)[0] == 1 &&
                      (*bias_shape)[1] == n)))) {
    return false;
  }

  Ort::ConstNode activation{nullptr};
  std::string activation_name;
  auto linear_consumers = linear_output.GetConsumers();
  if (linear_consumers.size() == 1 && linear_consumers[0].index == 0 &&
      IsOnnxOp(linear_consumers[0].node, "Relu")) {
    activation = linear_consumers[0].node;
    if (accepted_node_ids.count(activation.GetId()) != 0) {
      return false;
    }
    activation_name = "Relu";
  } else if (graph_output_names.count(Name(linear_output)) == 0 &&
             linear_consumers.empty()) {
    return false;
  }

  branch = {matmul, add, activation};
  group_key = std::string("Linear|") + Name(matmul_inputs[0]) + "|" +
              std::to_string((*weight_shape)[0]) + "|" +
              std::to_string((*weight_shape)[1]) + "|" + activation_name;
  return true;
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindParallelLinearFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::map<std::string, std::vector<ParallelLinearBranch>> groups;
  for (Ort::ConstNode node : all_nodes) {
    ParallelLinearBranch branch;
    std::string key;
    if (ParseBranch(node, graph_output_names, accepted_node_ids, branch, key)) {
      groups[key].push_back(branch);
    }
  }

  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (auto& [key, branches] : groups) {
    (void)key;
    if (branches.size() < 2) {
      continue;
    }
    std::vector<Ort::ConstNode> nodes;
    nodes.reserve(branches.size() * 3);
    for (const auto& branch : branches) {
      nodes.push_back(branch.linear);
      if (branch.add) {
        nodes.push_back(branch.add);
      }
      if (branch.activation) {
        nodes.push_back(branch.activation);
      }
    }
    fusions.push_back(std::move(nodes));
  }
  return fusions;
}

}  // namespace musa_ep
