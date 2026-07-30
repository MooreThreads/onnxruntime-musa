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

bool IsLinearActivationNode(Ort::ConstNode node) {
  return IsOnnxOp(node, "Relu") || IsOnnxOp(node, "LeakyRelu") ||
         IsOnnxOp(node, "Tanh") || IsOnnxOp(node, "Sigmoid");
}

bool IsBiasShapeForMatMulN(const std::vector<int64_t>& bias_shape, int64_t n) {
  if (n <= 0) {
    return true;
  }
  return (bias_shape.size() == 1 && bias_shape[0] == n) ||
         (bias_shape.size() == 2 && bias_shape[0] == 1 && bias_shape[1] == n);
}

bool CanFuseMatMulAddActivation(Ort::ConstNode matmul_node,
                                Ort::ConstNode add_node,
                                Ort::ConstNode activation_node,
                                int64_t add_matmul_input_idx) {
  std::vector<Ort::ConstValueInfo> matmul_inputs = matmul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> matmul_outputs = matmul_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> activation_inputs =
      activation_node.GetInputs();
  std::vector<Ort::ConstValueInfo> activation_outputs =
      activation_node.GetOutputs();
  if (matmul_inputs.size() != 2 || matmul_outputs.size() != 1 ||
      add_inputs.size() != 2 || add_outputs.size() != 1 ||
      activation_inputs.size() != 1 || activation_outputs.size() != 1 ||
      (add_matmul_input_idx != 0 && add_matmul_input_idx != 1)) {
    return false;
  }

  if (Name(matmul_outputs[0]) !=
          Name(add_inputs[static_cast<size_t>(add_matmul_input_idx)]) ||
      Name(add_outputs[0]) != Name(activation_inputs[0])) {
    return false;
  }

  const size_t bias_idx = static_cast<size_t>(1 - add_matmul_input_idx);
  if (!IsFloatTensorValueInfo(matmul_inputs[0]) ||
      !IsFloatTensorValueInfo(matmul_inputs[1]) ||
      !IsFloatTensorValueInfo(add_inputs[bias_idx]) ||
      !IsFloatTensorValueInfo(activation_outputs[0])) {
    return false;
  }

  auto b_shape = GetStaticShape(matmul_inputs[1]);
  if (b_shape.has_value() && b_shape->size() != 2) {
    return false;
  }

  auto a_shape = GetStaticShape(matmul_inputs[0]);
  if (a_shape.has_value() && a_shape->size() < 2) {
    return false;
  }

  if (a_shape.has_value() && b_shape.has_value() &&
      a_shape->back() != (*b_shape)[0]) {
    return false;
  }

  auto bias_shape = GetStaticShape(add_inputs[bias_idx]);
  if (bias_shape.has_value() && b_shape.has_value() &&
      !IsBiasShapeForMatMulN(*bias_shape, (*b_shape)[1])) {
    return false;
  }

  return true;
}

bool CanFuseMatMulAdd(Ort::ConstNode matmul_node, Ort::ConstNode add_node,
                      int64_t add_matmul_input_idx) {
  std::vector<Ort::ConstValueInfo> matmul_inputs = matmul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> matmul_outputs = matmul_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  if (matmul_inputs.size() != 2 || matmul_outputs.size() != 1 ||
      add_inputs.size() != 2 || add_outputs.size() != 1 ||
      (add_matmul_input_idx != 0 && add_matmul_input_idx != 1)) {
    return false;
  }

  if (Name(matmul_outputs[0]) !=
      Name(add_inputs[static_cast<size_t>(add_matmul_input_idx)])) {
    return false;
  }

  const size_t bias_idx = static_cast<size_t>(1 - add_matmul_input_idx);
  if (!IsFloatTensorValueInfo(matmul_inputs[0]) ||
      !IsFloatTensorValueInfo(matmul_inputs[1]) ||
      !IsFloatTensorValueInfo(add_inputs[bias_idx]) ||
      !IsFloatTensorValueInfo(add_outputs[0])) {
    return false;
  }

  auto b_shape = GetStaticShape(matmul_inputs[1]);
  if (!b_shape.has_value() || b_shape->size() != 2) {
    return false;
  }

  auto a_shape = GetStaticShape(matmul_inputs[0]);
  if (a_shape.has_value() && a_shape->size() < 2) {
    return false;
  }

  if (a_shape.has_value() && b_shape.has_value() &&
      a_shape->back() != (*b_shape)[0]) {
    return false;
  }

  auto bias_shape = GetStaticShape(add_inputs[bias_idx]);
  if (bias_shape.has_value() &&
      !IsBiasShapeForMatMulN(*bias_shape, (*b_shape)[1])) {
    return false;
  }

  return true;
}

bool CanFuseGemmActivation(Ort::ConstNode gemm_node,
                           Ort::ConstNode activation_node) {
  std::vector<Ort::ConstValueInfo> gemm_inputs = gemm_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gemm_outputs = gemm_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> activation_inputs =
      activation_node.GetInputs();
  std::vector<Ort::ConstValueInfo> activation_outputs =
      activation_node.GetOutputs();
  if ((gemm_inputs.size() != 2 && gemm_inputs.size() != 3) ||
      gemm_outputs.size() != 1 || activation_inputs.size() != 1 ||
      activation_outputs.size() != 1 ||
      Name(gemm_outputs[0]) != Name(activation_inputs[0])) {
    return false;
  }

  if (!IsFloatTensorValueInfo(gemm_inputs[0]) ||
      !IsFloatTensorValueInfo(gemm_inputs[1]) ||
      !IsFloatTensorValueInfo(activation_outputs[0])) {
    return false;
  }
  if (gemm_inputs.size() == 3 && !IsFloatTensorValueInfo(gemm_inputs[2])) {
    return false;
  }

  auto a_shape = GetStaticShape(gemm_inputs[0]);
  auto b_shape = GetStaticShape(gemm_inputs[1]);
  if (a_shape.has_value() && a_shape->size() != 2) {
    return false;
  }
  if (b_shape.has_value() && b_shape->size() != 2) {
    return false;
  }
  return true;
}

bool HasSameStaticElementCount(Ort::ConstValueInfo input,
                               Ort::ConstValueInfo output) {
  auto input_shape = GetStaticShape(input);
  auto output_shape = GetStaticShape(output);
  if (!input_shape.has_value() || !output_shape.has_value()) {
    return false;
  }
  int64_t input_elements = 1;
  int64_t output_elements = 1;
  for (int64_t dim : *input_shape) input_elements *= dim;
  for (int64_t dim : *output_shape) output_elements *= dim;
  return input_elements == output_elements;
}

bool IsAliasOnlyReshape(Ort::ConstNode reshape_node,
                        Ort::ConstValueInfo& data_input,
                        Ort::ConstValueInfo& data_output) {
  if (!IsOnnxOp(reshape_node, "Reshape")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> inputs = reshape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> outputs = reshape_node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1 ||
      !IsSmallIntegerInitializer(inputs[1]) ||
      !IsFloatTensorValueInfo(inputs[0]) ||
      !IsFloatTensorValueInfo(outputs[0]) ||
      !HasSameStaticElementCount(inputs[0], outputs[0])) {
    return false;
  }
  data_input = inputs[0];
  data_output = outputs[0];
  return true;
}

bool IsAliasOnlyUnsqueeze(Ort::ConstNode unsqueeze_node,
                          Ort::ConstValueInfo& data_input,
                          Ort::ConstValueInfo& data_output) {
  if (!IsOnnxOp(unsqueeze_node, "Unsqueeze")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> inputs = unsqueeze_node.GetInputs();
  std::vector<Ort::ConstValueInfo> outputs = unsqueeze_node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1 ||
      !ReadUnsqueezeAxes(unsqueeze_node).has_value() ||
      !IsFloatTensorValueInfo(inputs[0]) ||
      !IsFloatTensorValueInfo(outputs[0]) ||
      !HasSameStaticElementCount(inputs[0], outputs[0])) {
    return false;
  }
  data_input = inputs[0];
  data_output = outputs[0];
  return true;
}

// Match the ORT MatMulAddFusion lowering for a high-rank linear layer:
// [optional alias Reshape] -> Gemm -> alias Reshape -> activation
//                              -> [optional alias Reshape / Unsqueeze].
// The two Reshapes around Gemm only change tensor metadata; keeping them in
// the fused node lets GEMM run on its 2-D view and write the final logical
// output shape directly.
std::vector<std::vector<Ort::ConstNode>> FindDirectGemmActivationFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);

std::vector<std::vector<Ort::ConstNode>> FindGemmActivationFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  std::unordered_set<size_t> locally_selected = accepted_node_ids;

  for (Ort::ConstNode gemm_node : all_nodes) {
    if (!IsOnnxOp(gemm_node, "Gemm") ||
        locally_selected.count(gemm_node.GetId()) != 0) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> gemm_inputs = gemm_node.GetInputs();
    std::vector<Ort::ConstValueInfo> gemm_outputs = gemm_node.GetOutputs();
    if ((gemm_inputs.size() != 2 && gemm_inputs.size() != 3) ||
        gemm_outputs.size() != 1 ||
        graph_output_names.count(Name(gemm_outputs[0])) != 0) {
      continue;
    }

    Ort::ConstValueInfo gemm_input = gemm_inputs[0];
    Ort::ConstNode input_reshape{nullptr};
    Ort::ConstNode candidate_input_reshape{nullptr};
    if (GetProducer(gemm_input, candidate_input_reshape)) {
      Ort::ConstValueInfo reshape_input{nullptr};
      Ort::ConstValueInfo reshape_output{nullptr};
      if (IsAliasOnlyReshape(candidate_input_reshape, reshape_input,
                             reshape_output) &&
          Name(reshape_output) == Name(gemm_input) &&
          locally_selected.count(candidate_input_reshape.GetId()) == 0 &&
          HasOnlyConsumer(reshape_output, gemm_node, 0)) {
        auto source_shape = GetStaticShape(reshape_input);
        auto gemm_shape = GetStaticShape(gemm_input);
        if (source_shape.has_value() && source_shape->size() >= 2 &&
            gemm_shape.has_value() && gemm_shape->size() == 2) {
          input_reshape = candidate_input_reshape;
        }
      }
    }

    Ort::ConstValueInfo gemm_output = gemm_outputs[0];
    std::vector<Ort::ValueInfoConsumerProducerInfo> gemm_consumers =
        gemm_output.GetConsumers();
    if (gemm_consumers.size() != 1 || gemm_consumers[0].index != 0) {
      continue;
    }
    Ort::ConstNode output_reshape = gemm_consumers[0].node;
    Ort::ConstValueInfo reshape_input{nullptr};
    Ort::ConstValueInfo reshape_output{nullptr};
    if (!IsAliasOnlyReshape(output_reshape, reshape_input, reshape_output) ||
        Name(reshape_input) != Name(gemm_output) ||
        locally_selected.count(output_reshape.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> reshape_consumers =
        reshape_output.GetConsumers();
    if (reshape_consumers.size() != 1 || reshape_consumers[0].index != 0) {
      continue;
    }
    Ort::ConstNode activation_node = reshape_consumers[0].node;
    if (!IsLinearActivationNode(activation_node) ||
        locally_selected.count(activation_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> activation_outputs =
        activation_node.GetOutputs();
    if (activation_outputs.size() != 1) {
      continue;
    }
    Ort::ConstNode tail_node{nullptr};
    Ort::ConstValueInfo final_output = activation_outputs[0];
    if (graph_output_names.count(Name(final_output)) == 0) {
      std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
          final_output.GetConsumers();
      if (consumers.size() == 1 && consumers[0].index == 0) {
        Ort::ConstNode candidate_tail = consumers[0].node;
        Ort::ConstValueInfo tail_input{nullptr};
        Ort::ConstValueInfo tail_output{nullptr};
        if (locally_selected.count(candidate_tail.GetId()) == 0 &&
            ((IsAliasOnlyReshape(candidate_tail, tail_input, tail_output) ||
              IsAliasOnlyUnsqueeze(candidate_tail, tail_input, tail_output)) &&
             Name(tail_input) == Name(final_output))) {
          tail_node = candidate_tail;
          final_output = tail_output;
        }
      }
    }

    // The activation is intentionally separated from Gemm by Reshape, so
    // validate the GEMM inputs without the direct-adjacency requirement used
    // by CanFuseGemmActivation.
    auto a_shape = GetStaticShape(gemm_inputs[0]);
    auto b_shape = GetStaticShape(gemm_inputs[1]);
    if (!IsFloatTensorValueInfo(gemm_inputs[0]) ||
        !IsFloatTensorValueInfo(gemm_inputs[1]) ||
        (gemm_inputs.size() == 3 && !IsFloatTensorValueInfo(gemm_inputs[2])) ||
        !IsFloatTensorValueInfo(final_output) || !a_shape.has_value() ||
        a_shape->size() != 2 || !b_shape.has_value() || b_shape->size() != 2) {
      continue;
    }

    std::vector<Ort::ConstNode> fusion_nodes;
    if (input_reshape) fusion_nodes.push_back(input_reshape);
    fusion_nodes.push_back(gemm_node);
    fusion_nodes.push_back(output_reshape);
    fusion_nodes.push_back(activation_node);
    if (tail_node) fusion_nodes.push_back(tail_node);
    for (Ort::ConstNode node : fusion_nodes) {
      locally_selected.insert(node.GetId());
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  auto direct_fusions = FindDirectGemmActivationFusions(
      all_nodes, graph_output_names, locally_selected);
  for (auto& fusion : direct_fusions) {
    fusions.push_back(std::move(fusion));
  }
  return fusions;
}

std::vector<std::vector<Ort::ConstNode>> FindDirectGemmActivationFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode gemm_node : all_nodes) {
    if (!IsOnnxOp(gemm_node, "Gemm") ||
        accepted_node_ids.count(gemm_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> gemm_outputs = gemm_node.GetOutputs();
    if (gemm_outputs.size() != 1 ||
        graph_output_names.count(gemm_outputs[0].GetName()) != 0) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> gemm_consumers =
        gemm_outputs[0].GetConsumers();
    if (gemm_consumers.size() != 1 || gemm_consumers[0].index != 0) {
      continue;
    }

    Ort::ConstNode activation_node = gemm_consumers[0].node;
    if (!IsLinearActivationNode(activation_node) ||
        accepted_node_ids.count(activation_node.GetId()) != 0) {
      continue;
    }

    if (!CanFuseGemmActivation(gemm_node, activation_node)) {
      continue;
    }

    fusions.push_back({gemm_node, activation_node});
  }

  return fusions;
}

std::vector<std::vector<Ort::ConstNode>> FindFusedGemmFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode matmul_node : all_nodes) {
    if (!IsOnnxOp(matmul_node, "MatMul") ||
        accepted_node_ids.count(matmul_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> matmul_outputs = matmul_node.GetOutputs();
    if (matmul_outputs.size() != 1 ||
        graph_output_names.count(matmul_outputs[0].GetName()) != 0) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> matmul_consumers =
        matmul_outputs[0].GetConsumers();
    if (matmul_consumers.size() != 1 || matmul_consumers[0].index < 0 ||
        matmul_consumers[0].index > 1) {
      continue;
    }

    Ort::ConstNode add_node = matmul_consumers[0].node;
    if (!IsOnnxOp(add_node, "Add") ||
        accepted_node_ids.count(add_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
    if (add_outputs.size() != 1 ||
        graph_output_names.count(add_outputs[0].GetName()) != 0) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> add_consumers =
        add_outputs[0].GetConsumers();
    if (add_consumers.size() == 1 && add_consumers[0].index == 0) {
      Ort::ConstNode activation_node = add_consumers[0].node;
      if (IsLinearActivationNode(activation_node) &&
          accepted_node_ids.count(activation_node.GetId()) == 0 &&
          CanFuseMatMulAddActivation(matmul_node, add_node, activation_node,
                                     matmul_consumers[0].index)) {
        fusions.push_back({matmul_node, add_node, activation_node});
        continue;
      }
    }

    if (!CanFuseMatMulAdd(matmul_node, add_node, matmul_consumers[0].index)) {
      continue;
    }

    fusions.push_back({matmul_node, add_node});
  }

  return fusions;
}

}  // namespace musa_ep
