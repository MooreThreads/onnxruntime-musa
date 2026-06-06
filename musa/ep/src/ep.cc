// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include <array>
#include <cassert>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ep_factory.h"
#include "ep_profiling.h"
#include "fusion/concat_matmul_fusion.h"
#include "fusion/fusion_node_compute.h"
#include "fusion/linear_fusion.h"
#include "plugin_ep_utils.h"

namespace {

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(const Ort::ConstNode& node, const char* op_type) {
  return node.GetOperatorType() == op_type && IsOnnxDomain(node.GetDomain());
}

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
}

bool IsFloatTensorValueInfo(Ort::ConstValueInfo value_info) {
  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }

  auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  return type_shape.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

std::optional<std::vector<int64_t>> GetStaticShape(
    Ort::ConstValueInfo value_info) {
  auto shape = GetTensorShape(value_info);
  if (!shape.has_value()) {
    return std::nullopt;
  }

  for (int64_t dim : *shape) {
    if (dim <= 0) {
      return std::nullopt;
    }
  }

  return shape;
}

std::optional<int64_t> GetIntAttribute(Ort::ConstNode node,
                                       const std::string& name) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK()) {
    return std::nullopt;
  }

  int64_t value = 0;
  status = attr.GetValue(value);
  if (!status.IsOK()) {
    return std::nullopt;
  }

  return value;
}

bool NormalizeAxis(int64_t axis, size_t rank, int64_t& normalized_axis) {
  const int64_t signed_rank = static_cast<int64_t>(rank);
  if (rank == 0 || axis < -signed_rank || axis >= signed_rank) {
    return false;
  }

  normalized_axis = axis < 0 ? axis + signed_rank : axis;
  return true;
}

constexpr int64_t kSmallInitializerThreshold = 100;

bool MemTypeOnCpuExplicitly(OrtMemType mem_type) {
  return mem_type == OrtMemTypeCPUInput || mem_type == OrtMemTypeCPUOutput;
}

std::vector<Ort::ConstNode> GetOutputNodes(
    const std::vector<Ort::ConstValueInfo>& node_outputs) {
  std::vector<Ort::ConstNode> output_nodes;
  for (Ort::ConstValueInfo output : node_outputs) {
    if (output == nullptr) {
      continue;
    }
    for (const auto& consumer : output.GetConsumers()) {
      output_nodes.push_back(consumer.node);
    }
  }
  return output_nodes;
}

bool IsCpuPreferredAllowedType(Ort::ConstValueInfo input) {
  Ort::ConstTypeInfo type_info = input.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }

  auto type_shape_info = type_info.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType elem_type = type_shape_info.GetElementType();
  return elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
         elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 &&
         elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN &&
         elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ &&
         elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 &&
         elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ &&
         elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1;
}

bool IsSmallInitializer(Ort::ConstValueInfo input) {
  if (!input.IsConstantInitializer()) {
    return false;
  }
  Ort::ConstTypeInfo type_info = input.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  return type_info.GetTensorTypeAndShapeInfo().GetElementCount() <=
         kSmallInitializerThreshold;
}

OrtStatus* GetCpuPreferredNodes(
    const OrtGraph& ort_graph, OrtEpGraphSupportInfo& graph_support_info,
    const OrtEpApi& ep_api, std::span<const OrtNode* const> tentative_nodes,
    std::unordered_set<const OrtNode*>& cpu_preferred_nodes) noexcept {
  try {
    const OrtApi& ort_api = Ort::GetApi();
    Ort::ConstGraph graph{&ort_graph};
    std::vector<Ort::ConstNode> ordered_nodes = graph.GetNodes();
    if (ordered_nodes.empty()) {
      return nullptr;
    }

    std::unordered_map<size_t, Ort::ConstNode> node_id_to_node;
    std::unordered_map<size_t, size_t> node_id_to_order_map;
    for (size_t i = 0; i < ordered_nodes.size(); ++i) {
      size_t node_id = ordered_nodes[i].GetId();
      node_id_to_node.emplace(node_id, ordered_nodes[i]);
      node_id_to_order_map.emplace(node_id, i);
    }

    auto greater_order_comp = [&](size_t lhs, size_t rhs) {
      return node_id_to_order_map[lhs] > node_id_to_order_map[rhs];
    };
    std::priority_queue<size_t, std::vector<size_t>,
                        decltype(greater_order_comp)>
        candidates(greater_order_comp);
    std::unordered_set<const OrtValueInfo*> cpu_output_args;
    std::unordered_set<size_t> provider_nodes;
    provider_nodes.reserve(tentative_nodes.size());
    std::unordered_map<size_t, Ort::ConstKernelDef> node_to_kernel;
    node_to_kernel.reserve(tentative_nodes.size());

    for (const OrtNode* ort_node : tentative_nodes) {
      Ort::ConstNode node{ort_node};
      size_t node_id = node.GetId();
      provider_nodes.insert(node_id);

      const OrtKernelDef* ort_kernel_def = nullptr;
      RETURN_IF_ERROR(ep_api.EpGraphSupportInfo_LookUpKernel(
          &graph_support_info, node, &ort_kernel_def));
      RETURN_IF(ort_kernel_def == nullptr, ort_api,
                "Missing kernel definition for tentative MUSA node");

      Ort::ConstKernelDef kernel_def{ort_kernel_def};
      node_to_kernel.emplace(node_id, kernel_def);

      std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
      for (size_t out_index = 0; out_index < outputs.size(); ++out_index) {
        Ort::ConstValueInfo output = outputs[out_index];
        if (output == nullptr) {
          continue;
        }
        if (MemTypeOnCpuExplicitly(kernel_def.GetOutputMemType(out_index))) {
          cpu_output_args.insert(output);
          for (const auto& consumer : output.GetConsumers()) {
            candidates.push(consumer.node.GetId());
          }
        }
      }
    }

    std::unordered_set<size_t> visited;
    visited.reserve(candidates.size());
    std::unordered_set<const OrtNode*> cpu_nodes;
    cpu_nodes.reserve(candidates.size());

    while (!candidates.empty()) {
      size_t cur = candidates.top();
      candidates.pop();
      if (!visited.insert(cur).second) {
        continue;
      }

      auto node_iter = node_id_to_node.find(cur);
      RETURN_IF(node_iter == node_id_to_node.end(), ort_api,
                "Unable to get node while finding CPU-preferred subgraph");
      Ort::ConstNode node = node_iter->second;

      if (provider_nodes.find(cur) == provider_nodes.end()) {
        std::string ep_name = node.GetEpName();
        if (ep_name.empty() || ep_name == "CPUExecutionProvider") {
          std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
          for (Ort::ConstValueInfo output : outputs) {
            if (output != nullptr) {
              cpu_output_args.insert(output);
            }
          }
          for (Ort::ConstNode downstream_node : GetOutputNodes(outputs)) {
            candidates.push(downstream_node.GetId());
          }
        }
        continue;
      }

      auto kernel_iter = node_to_kernel.find(cur);
      RETURN_IF(kernel_iter == node_to_kernel.end(), ort_api,
                "Unable to get kernel definition for CPU-preferred node");

      bool place_in_cpu = true;
      std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
      for (size_t i = 0; i < inputs.size(); ++i) {
        Ort::ConstValueInfo input = inputs[i];
        if (input == nullptr) {
          continue;
        }
        if (!IsCpuPreferredAllowedType(input)) {
          place_in_cpu = false;
          break;
        }
        if (IsSmallInitializer(input) || input.IsRequiredGraphInput() ||
            input.IsOptionalGraphInput()) {
          continue;
        }
        if (cpu_output_args.find(input) == cpu_output_args.end()) {
          place_in_cpu = false;
          break;
        }
        if (MemTypeOnCpuExplicitly(kernel_iter->second.GetInputMemType(i))) {
          place_in_cpu = false;
          break;
        }
      }

      if (place_in_cpu) {
        cpu_nodes.insert(node);
        std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
        for (Ort::ConstValueInfo output : outputs) {
          if (output != nullptr) {
            cpu_output_args.insert(output);
          }
        }
        for (Ort::ConstNode downstream_node : GetOutputNodes(outputs)) {
          candidates.push(downstream_node.GetId());
        }
      }
    }

    cpu_preferred_nodes = std::move(cpu_nodes);
    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  } catch (...) {
    Ort::Status status("Unknown exception", ORT_EP_FAIL);
    return status.release();
  }
}

bool CanFuseConcatMatMul(Ort::ConstNode concat_node, Ort::ConstNode matmul_node,
                         int64_t concat_input_idx) {
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> matmul_inputs = matmul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> matmul_outputs = matmul_node.GetOutputs();
  if (concat_inputs.size() < 2 || concat_outputs.size() != 1 ||
      matmul_inputs.size() != 2 || matmul_outputs.size() != 1 ||
      (concat_input_idx != 0 && concat_input_idx != 1)) {
    return false;
  }

  const std::string concat_output_name = concat_outputs[0].GetName();
  if (matmul_inputs[static_cast<size_t>(concat_input_idx)].GetName() !=
      concat_output_name) {
    return false;
  }

  if (!IsFloatTensorValueInfo(matmul_outputs[0]) ||
      !IsFloatTensorValueInfo(
          matmul_inputs[static_cast<size_t>(1 - concat_input_idx)])) {
    return false;
  }

  for (Ort::ConstValueInfo input : concat_inputs) {
    if (!IsFloatTensorValueInfo(input)) {
      return false;
    }
  }

  auto axis_attr = GetIntAttribute(concat_node, "axis");
  if (!axis_attr.has_value()) {
    return false;
  }

  // The fused implementation only supports equal-rank MatMul inputs. Require
  // enough static shape information to prove that before claiming capability.
  auto first_concat_shape = GetStaticShape(concat_inputs[0]);
  if (!first_concat_shape.has_value()) {
    return false;
  }
  if (first_concat_shape->size() < 2) {
    return false;
  }

  int64_t axis = 0;
  if (!NormalizeAxis(*axis_attr, first_concat_shape->size(), axis)) {
    return false;
  }

  std::vector<int64_t> concat_shape = *first_concat_shape;
  concat_shape[static_cast<size_t>(axis)] = 0;
  bool all_concat_shapes_known = true;
  for (Ort::ConstValueInfo input : concat_inputs) {
    auto shape = GetStaticShape(input);
    if (!shape.has_value()) {
      all_concat_shapes_known = false;
      continue;
    }
    if (shape->size() != concat_shape.size()) {
      return false;
    }

    for (size_t dim = 0; dim < shape->size(); ++dim) {
      if (dim == static_cast<size_t>(axis)) {
        continue;
      }
      if ((*shape)[dim] != concat_shape[dim]) {
        return false;
      }
    }
    concat_shape[static_cast<size_t>(axis)] +=
        (*shape)[static_cast<size_t>(axis)];
  }

  auto other_shape =
      GetStaticShape(matmul_inputs[static_cast<size_t>(1 - concat_input_idx)]);
  if (!other_shape.has_value() || !all_concat_shapes_known) {
    return false;
  }
  if (other_shape->size() != concat_shape.size()) {
    return false;
  }

  const std::vector<int64_t>& lhs_shape =
      concat_input_idx == 0 ? concat_shape : *other_shape;
  const std::vector<int64_t>& rhs_shape =
      concat_input_idx == 0 ? *other_shape : concat_shape;
  const size_t rank = lhs_shape.size();
  for (size_t dim = 0; dim + 2 < rank; ++dim) {
    if (lhs_shape[dim] != rhs_shape[dim]) {
      return false;
    }
  }
  return lhs_shape[rank - 1] == rhs_shape[rank - 2];
}

std::vector<std::vector<Ort::ConstNode>> FindConcatMatMulFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode concat_node : all_nodes) {
    if (!IsOnnxOp(concat_node, "Concat") ||
        fused_node_ids.count(concat_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
    if (concat_outputs.size() != 1 ||
        graph_output_names.count(concat_outputs[0].GetName()) != 0) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
        concat_outputs[0].GetConsumers();
    if (consumers.size() != 1 || consumers[0].index < 0 ||
        consumers[0].index > 1) {
      continue;
    }

    Ort::ConstNode matmul_node = consumers[0].node;
    if (!IsOnnxOp(matmul_node, "MatMul") ||
        fused_node_ids.count(matmul_node.GetId()) != 0) {
      continue;
    }

    if (!CanFuseConcatMatMul(concat_node, matmul_node, consumers[0].index)) {
      continue;
    }

    fusions.push_back({concat_node, matmul_node});
    fused_node_ids.insert(concat_node.GetId());
    fused_node_ids.insert(matmul_node.GetId());
  }

  return fusions;
}

bool IsLinearActivationNode(Ort::ConstNode node) {
  return IsOnnxOp(node, "Relu") || IsOnnxOp(node, "LeakyRelu") ||
         IsOnnxOp(node, "Tanh");
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

std::vector<std::vector<Ort::ConstNode>> FindGemmActivationFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode gemm_node : all_nodes) {
    if (!IsOnnxOp(gemm_node, "Gemm") ||
        fused_node_ids.count(gemm_node.GetId()) != 0) {
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
        fused_node_ids.count(activation_node.GetId()) != 0) {
      continue;
    }

    if (!CanFuseGemmActivation(gemm_node, activation_node)) {
      continue;
    }

    fusions.push_back({gemm_node, activation_node});
    fused_node_ids.insert(gemm_node.GetId());
    fused_node_ids.insert(activation_node.GetId());
  }

  return fusions;
}

std::vector<std::vector<Ort::ConstNode>> FindFusedGemmFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode matmul_node : all_nodes) {
    if (!IsOnnxOp(matmul_node, "MatMul") ||
        fused_node_ids.count(matmul_node.GetId()) != 0) {
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
        fused_node_ids.count(add_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
    if (add_outputs.size() != 1 ||
        graph_output_names.count(add_outputs[0].GetName()) != 0) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> add_consumers =
        add_outputs[0].GetConsumers();
    if (add_consumers.size() != 1 || add_consumers[0].index != 0) {
      continue;
    }

    Ort::ConstNode activation_node = add_consumers[0].node;
    if (!IsLinearActivationNode(activation_node) ||
        fused_node_ids.count(activation_node.GetId()) != 0) {
      continue;
    }

    if (!CanFuseMatMulAddActivation(matmul_node, add_node, activation_node,
                                    matmul_consumers[0].index)) {
      continue;
    }

    fusions.push_back({matmul_node, add_node, activation_node});
    fused_node_ids.insert(matmul_node.GetId());
    fused_node_ids.insert(add_node.GetId());
    fused_node_ids.insert(activation_node.GetId());
  }

  return fusions;
}

}  // namespace

MusaEp::MusaEp(MusaEpFactory& factory, const Config& config,
               const OrtLogger& logger)
    : OrtEp{},  // explicitly call the struct ctor to ensure all optional values
                // are default initialized
      factory_{factory},
      ort_api_{factory.GetOrtApi()},
      ep_api_{factory.GetEpApi()},
      name_{factory.GetEpName()},
      config_{config},
      logger_{logger} {
  ort_version_supported =
      ORT_API_VERSION;  // set to the ORT version we were compiled with.

  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  CreateProfiler = CreateProfilerImpl;

  IGNORE_ORTSTATUS(ort_api_.Logger_LogMessage(
      &logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
      ("MUSAExecutionProvider has been created with name " + name_).c_str(),
      ORT_FILE, __LINE__, __FUNCTION__));
}

MusaEp::~MusaEp() = default;

/*static*/
const char* ORT_API_CALL MusaEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const MusaEp*>(this_ptr);
  return ep->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL
MusaEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                          OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    MusaEp* ep = static_cast<MusaEp*>(this_ptr);

    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> all_nodes = graph.GetNodes();

    if (all_nodes.empty()) {
      return nullptr;  // No nodes to process
    }

    std::unordered_set<std::string> graph_output_names;
    for (Ort::ConstValueInfo output : graph.GetOutputs()) {
      graph_output_names.insert(output.GetName());
    }

    std::unordered_set<size_t> fused_node_ids;
    std::vector<std::vector<Ort::ConstNode>> concat_matmul_fusions =
        FindConcatMatMulFusions(all_nodes, graph_output_names, fused_node_ids);

    for (const auto& fusion_nodes : concat_matmul_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }
    std::vector<std::vector<Ort::ConstNode>> gemm_activation_fusions =
        FindGemmActivationFusions(all_nodes, graph_output_names,
                                  fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> fused_gemm_fusions =
        FindFusedGemmFusions(all_nodes, graph_output_names, fused_node_ids);

    for (const auto& fusion_nodes : gemm_activation_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    for (const auto& fusion_nodes : fused_gemm_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    std::vector<Ort::ConstNode> candidate_nodes;
    std::vector<const OrtNode*> tentative_nodes;

    // Mark non-fused nodes as supported if we have a registered kernel. Defer
    // adding them until after the CUDA-style CPU-preferred shape subgraph pass.
    for (const auto& node : all_nodes) {
      if (fused_node_ids.count(node.GetId()) != 0) {
        continue;
      }

      std::string ep_name = node.GetEpName();
      if (!ep_name.empty()) {
        if (ep_name == ep->name_) {
          candidate_nodes.push_back(node);
          tentative_nodes.push_back(node);
        }
        continue;
      }

      const OrtKernelDef* kernel_def = nullptr;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_LookUpKernel(
          graph_support_info, node, &kernel_def));
      if (kernel_def != nullptr) {
        candidate_nodes.push_back(node);
        tentative_nodes.push_back(node);
      }
    }

    std::unordered_set<const OrtNode*> cpu_preferred_nodes;
    if (!tentative_nodes.empty()) {
      RETURN_IF_ERROR(GetCpuPreferredNodes(
          *ort_graph, *graph_support_info, ep->ep_api_,
          std::span<const OrtNode* const>(tentative_nodes.data(),
                                          tentative_nodes.size()),
          cpu_preferred_nodes));
    }

    for (const auto& node : candidate_nodes) {
      if (cpu_preferred_nodes.count(node) == 0) {
        RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddSingleNode(
            graph_support_info, node));
      }
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL MusaEp::GetKernelRegistryImpl(
    _In_ OrtEp* this_ptr, _Outptr_result_maybenull_ const OrtKernelRegistry**
                              kernel_registry) noexcept {
  MusaEp* ep = static_cast<MusaEp*>(this_ptr);

  *kernel_registry = nullptr;

  // Get the cached kernel registry from parent factory to avoid recreating the
  // kernel registry for every EP instance.
  RETURN_IF_ERROR(ep->factory_.GetKernelRegistryForEp(*ep, kernel_registry));
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL MusaEp::CreateProfilerImpl(
    OrtEp* this_ptr, OrtEpProfilerImpl** profiler) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  MusaEp* ep = static_cast<MusaEp*>(this_ptr);
  auto profiler_unique_ptr = std::make_unique<MusaEpProfiler>(ep->ep_api_);

  *profiler = profiler_unique_ptr.release();
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}
