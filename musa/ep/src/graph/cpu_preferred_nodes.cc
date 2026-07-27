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

#include "graph/cpu_preferred_nodes.h"

#include <queue>
#include <unordered_map>

#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

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

bool IsIntegerMetadataType(Ort::ConstValueInfo input) {
  Ort::ConstTypeInfo type_info = input.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }

  auto type_shape_info = type_info.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType elem_type = type_shape_info.GetElementType();
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

bool IsCpuPreferredMetadataType(Ort::ConstValueInfo input) {
  Ort::ConstTypeInfo type_info = input.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }

  auto type_shape_info = type_info.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType elem_type = type_shape_info.GetElementType();
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
}

bool IsCpuPreferredMetadataOp(Ort::ConstNode node) {
  if (IsOnnxOp(node, "Cast")) {
    auto cast_to = GetIntAttribute(node, "to");
    return cast_to.has_value() &&
           (*cast_to == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
            *cast_to == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  }

  if (IsOnnxOp(node, "ConstantOfShape")) {
    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
    return outputs.size() == 1 && IsIntegerMetadataType(outputs[0]);
  }

  return IsOnnxOp(node, "Gather") || IsOnnxOp(node, "Concat") ||
         IsOnnxOp(node, "Equal") || IsOnnxOp(node, "Where") ||
         IsOnnxOp(node, "Reshape") || IsOnnxOp(node, "Slice") ||
         IsOnnxOp(node, "Split") || IsOnnxOp(node, "Squeeze") ||
         IsOnnxOp(node, "Unsqueeze") || IsOnnxOp(node, "Add") ||
         IsOnnxOp(node, "Sub") || IsOnnxOp(node, "Mul") ||
         IsOnnxOp(node, "Div") || IsOnnxOp(node, "Max") ||
         IsOnnxOp(node, "Min") || IsOnnxOp(node, "ReduceProd") ||
         IsOnnxOp(node, "ReduceMax");
}

bool HasOnlyCpuPreferredMetadataOutputs(Ort::ConstNode node) {
  std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.empty()) {
    return false;
  }
  for (Ort::ConstValueInfo output : outputs) {
    if (output == nullptr || !IsCpuPreferredMetadataType(output)) {
      return false;
    }
  }
  return true;
}

bool CanKeepCpuPreferredOutputs(
    Ort::ConstNode node, const std::unordered_set<size_t>& provider_nodes,
    const std::unordered_map<size_t, Ort::ConstKernelDef>& node_to_kernel,
    std::unordered_map<size_t, bool>& cpu_preferred_downstream_cache,
    std::unordered_set<size_t>& visiting_cpu_preferred);

bool CanCpuPreferMetadataSubgraph(
    Ort::ConstNode node, const std::unordered_set<size_t>& provider_nodes,
    const std::unordered_map<size_t, Ort::ConstKernelDef>& node_to_kernel,
    std::unordered_map<size_t, bool>& cpu_preferred_downstream_cache,
    std::unordered_set<size_t>& visiting_cpu_preferred) {
  const size_t node_id = node.GetId();
  auto cached = cpu_preferred_downstream_cache.find(node_id);
  if (cached != cpu_preferred_downstream_cache.end()) {
    return cached->second;
  }
  if (!visiting_cpu_preferred.insert(node_id).second) {
    return false;
  }

  bool can_prefer = IsCpuPreferredMetadataOp(node) &&
                    HasOnlyCpuPreferredMetadataOutputs(node) &&
                    CanKeepCpuPreferredOutputs(
                        node, provider_nodes, node_to_kernel,
                        cpu_preferred_downstream_cache, visiting_cpu_preferred);
  visiting_cpu_preferred.erase(node_id);
  cpu_preferred_downstream_cache.emplace(node_id, can_prefer);
  return can_prefer;
}

bool CanConsumerReadCpuPreferredOutput(
    const Ort::ValueInfoConsumerProducerInfo& consumer,
    const std::unordered_set<size_t>& provider_nodes,
    const std::unordered_map<size_t, Ort::ConstKernelDef>& node_to_kernel,
    std::unordered_map<size_t, bool>& cpu_preferred_downstream_cache,
    std::unordered_set<size_t>& visiting_cpu_preferred) {
  Ort::ConstNode consumer_node = consumer.node;
  const size_t consumer_node_id = consumer_node.GetId();
  if (provider_nodes.find(consumer_node_id) == provider_nodes.end()) {
    return true;
  }

  auto kernel_iter = node_to_kernel.find(consumer_node_id);
  if (kernel_iter == node_to_kernel.end()) {
    return false;
  }
  if (consumer.index >= 0 &&
      MemTypeOnCpuExplicitly(kernel_iter->second.GetInputMemType(
          static_cast<size_t>(consumer.index)))) {
    return true;
  }
  return CanCpuPreferMetadataSubgraph(
      consumer_node, provider_nodes, node_to_kernel,
      cpu_preferred_downstream_cache, visiting_cpu_preferred);
}

bool CanKeepCpuPreferredOutputs(
    Ort::ConstNode node, const std::unordered_set<size_t>& provider_nodes,
    const std::unordered_map<size_t, Ort::ConstKernelDef>& node_to_kernel,
    std::unordered_map<size_t, bool>& cpu_preferred_downstream_cache,
    std::unordered_set<size_t>& visiting_cpu_preferred) {
  for (Ort::ConstValueInfo output : node.GetOutputs()) {
    if (output == nullptr) {
      continue;
    }
    for (const auto& consumer : output.GetConsumers()) {
      if (IsOnnxOp(node, "ConstantOfShape") &&
          IsOnnxOp(consumer.node, "Gather")) {
        return false;
      }
      if (!CanConsumerReadCpuPreferredOutput(
              consumer, provider_nodes, node_to_kernel,
              cpu_preferred_downstream_cache, visiting_cpu_preferred)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

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
    std::unordered_map<size_t, bool> cpu_preferred_downstream_cache;
    std::unordered_set<size_t> visiting_cpu_preferred;

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

      if (!CanCpuPreferMetadataSubgraph(node, provider_nodes, node_to_kernel,
                                        cpu_preferred_downstream_cache,
                                        visiting_cpu_preferred)) {
        continue;
      }

      bool place_in_cpu = true;
      bool has_cpu_metadata_input = false;
      std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
      for (size_t i = 0; i < inputs.size(); ++i) {
        Ort::ConstValueInfo input = inputs[i];
        if (input == nullptr) {
          continue;
        }
        if (!IsCpuPreferredMetadataType(input)) {
          place_in_cpu = false;
          break;
        }
        if (IsSmallInitializer(input)) {
          continue;
        }
        if (cpu_output_args.find(input) == cpu_output_args.end()) {
          place_in_cpu = false;
          break;
        }
        has_cpu_metadata_input = true;
        if (MemTypeOnCpuExplicitly(kernel_iter->second.GetInputMemType(i))) {
          place_in_cpu = false;
          break;
        }
      }

      if (!has_cpu_metadata_input) {
        place_in_cpu = false;
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

}  // namespace musa_ep
