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
  Ort::ConstValueInfo output{nullptr};
  ONNXTensorElementDataType elem_type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
};

bool IsParallelLinearStorageType(ONNXTensorElementDataType elem_type) {
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
}

bool RequireSameParallelLinearStorageType(
    const std::vector<Ort::ConstValueInfo>& values,
    ONNXTensorElementDataType& elem_type) {
  return RequireSameElementType(values, elem_type) &&
         IsParallelLinearStorageType(elem_type);
}

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
      matmul_outputs.size() != 1 || !matmul_inputs[1].IsConstantInitializer()) {
    return false;
  }
  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  if (!RequireSameParallelLinearStorageType(
          {matmul_inputs[0], matmul_inputs[1], matmul_outputs[0]}, elem_type)) {
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
          GetTensorElementType(candidate_bias) == elem_type &&
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
  if (has_bias && (GetTensorElementType(bias) != elem_type ||
                   !bias.IsConstantInitializer())) {
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

  if (activation) {
    linear_output = activation.GetOutputs()[0];
  }
  // The non-FP32 runtime path is deliberately a raw projection/split path.
  // Bias and activation require dtype-specific arithmetic and must remain on
  // the existing kernels until those epilogues are implemented.
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      (has_bias || activation)) {
    return false;
  }
  branch = {matmul, add, activation, linear_output, elem_type};
  group_key = std::string("Linear|") + Name(matmul_inputs[0]) + "|" +
              std::to_string((*weight_shape)[0]) + "|" +
              std::to_string((*weight_shape)[1]) + "|" + activation_name;
  return true;
}

bool IsExpectedInputPair(Ort::ConstNode node, Ort::ConstValueInfo lhs,
                         Ort::ConstValueInfo rhs) {
  auto inputs = node.GetInputs();
  return inputs.size() == 2 &&
         ((Name(inputs[0]) == Name(lhs) && Name(inputs[1]) == Name(rhs)) ||
          (Name(inputs[0]) == Name(rhs) && Name(inputs[1]) == Name(lhs)));
}

bool TryAppendGatedMlpNodes(
    const std::vector<ParallelLinearBranch>& branches,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& nodes) {
  if (branches.size() != 2 || branches[0].activation ||
      branches[1].activation) {
    return false;
  }

  for (size_t gate_index = 0; gate_index < branches.size(); ++gate_index) {
    const size_t up_index = 1 - gate_index;
    Ort::ConstValueInfo gate = branches[gate_index].output;
    Ort::ConstValueInfo up = branches[up_index].output;
    if (graph_output_names.count(Name(gate)) != 0 ||
        graph_output_names.count(Name(up)) != 0) {
      continue;
    }

    Ort::ConstNode sigmoid{nullptr};
    for (const auto& consumer : gate.GetConsumers()) {
      if (consumer.index == 0 && IsOnnxOp(consumer.node, "Sigmoid")) {
        sigmoid = consumer.node;
      }
    }
    if (!sigmoid || accepted_node_ids.count(sigmoid.GetId()) != 0) {
      continue;
    }
    auto sigmoid_inputs = sigmoid.GetInputs();
    auto sigmoid_outputs = sigmoid.GetOutputs();
    if (sigmoid_inputs.size() != 1 || sigmoid_outputs.size() != 1 ||
        Name(sigmoid_inputs[0]) != Name(gate) ||
        graph_output_names.count(Name(sigmoid_outputs[0])) != 0) {
      continue;
    }

    auto sigmoid_consumers = sigmoid_outputs[0].GetConsumers();
    if (sigmoid_consumers.size() != 1 ||
        !IsOnnxOp(sigmoid_consumers[0].node, "Mul") ||
        accepted_node_ids.count(sigmoid_consumers[0].node.GetId()) != 0) {
      continue;
    }
    Ort::ConstNode gate_mul = sigmoid_consumers[0].node;
    auto gate_mul_outputs = gate_mul.GetOutputs();
    if (gate_mul_outputs.size() != 1 ||
        !IsExpectedInputPair(gate_mul, gate, sigmoid_outputs[0]) ||
        graph_output_names.count(Name(gate_mul_outputs[0])) != 0) {
      continue;
    }

    auto gate_consumers = gate.GetConsumers();
    if (gate_consumers.size() != 2) {
      continue;
    }
    bool has_sigmoid_consumer = false;
    bool has_gate_mul_consumer = false;
    for (const auto& consumer : gate_consumers) {
      has_sigmoid_consumer |= consumer.node.GetId() == sigmoid.GetId();
      has_gate_mul_consumer |= consumer.node.GetId() == gate_mul.GetId();
    }
    if (!has_sigmoid_consumer || !has_gate_mul_consumer) {
      continue;
    }

    auto gate_mul_consumers = gate_mul_outputs[0].GetConsumers();
    auto up_consumers = up.GetConsumers();
    if (gate_mul_consumers.size() != 1 || up_consumers.size() != 1 ||
        gate_mul_consumers[0].node.GetId() != up_consumers[0].node.GetId() ||
        !IsOnnxOp(gate_mul_consumers[0].node, "Mul")) {
      continue;
    }
    Ort::ConstNode output_mul = gate_mul_consumers[0].node;
    auto output_mul_outputs = output_mul.GetOutputs();
    if (accepted_node_ids.count(output_mul.GetId()) != 0 ||
        output_mul_outputs.size() != 1 ||
        !IsExpectedInputPair(output_mul, gate_mul_outputs[0], up)) {
      continue;
    }

    nodes.push_back(sigmoid);
    nodes.push_back(gate_mul);
    nodes.push_back(output_mul);
    return true;
  }
  return false;
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
    if (branches[0].elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        branches.size() > 3) {
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
    if (branches[0].elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      TryAppendGatedMlpNodes(branches, graph_output_names, accepted_node_ids,
                             nodes);
    }
    fusions.push_back(std::move(nodes));
  }
  return fusions;
}

}  // namespace musa_ep
