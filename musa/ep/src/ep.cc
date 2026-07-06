// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ep_factory.h"
#include "ep_profiling.h"
#include "ep_stream.h"
#include "fusion/centered_reduce_fusion.h"
#include "fusion/concat_matmul_fusion.h"
#include "fusion/fusion_node_compute.h"
#include "fusion/linear_fusion.h"
#include "fusion/shape_reshape_fusion.h"
#include "fusion/split_reduce_fusion.h"
#include "fusion/tile_concat_fusion.h"
#include "graph_mermaid_dump.h"
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

std::optional<std::vector<int64_t>> GetIntsAttribute(Ort::ConstNode node,
                                                     const std::string& name) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK()) {
    return std::nullopt;
  }

  std::vector<int64_t> values;
  status = attr.GetValueArray<int64_t>(values);
  if (!status.IsOK()) {
    return std::nullopt;
  }
  return values;
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

std::optional<std::vector<int64_t>> ReadSmallIntInitializer(
    Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    return std::nullopt;
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    return std::nullopt;
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  const size_t count = static_cast<size_t>(info.GetElementCount());
  if (count > kSmallInitializerThreshold) {
    return std::nullopt;
  }

  std::vector<int64_t> result;
  result.reserve(count);
  ONNXTensorElementDataType elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t* data = value.GetTensorData<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      result.push_back(static_cast<int64_t>(data[i]));
    }
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* data = value.GetTensorData<int64_t>();
    result.assign(data, data + count);
  } else {
    return std::nullopt;
  }
  return result;
}

std::optional<std::vector<int64_t>> ReadIntInitializerNoLimit(
    Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    return std::nullopt;
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    return std::nullopt;
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  const size_t count = static_cast<size_t>(info.GetElementCount());
  std::vector<int64_t> result;
  result.reserve(count);
  ONNXTensorElementDataType elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t* data = value.GetTensorData<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      result.push_back(static_cast<int64_t>(data[i]));
    }
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* data = value.GetTensorData<int64_t>();
    result.assign(data, data + count);
  } else {
    return std::nullopt;
  }
  return result;
}

std::optional<std::vector<int64_t>> ReadUnsqueezeAxes(
    Ort::ConstNode unsqueeze_node) {
  std::vector<Ort::ConstValueInfo> inputs = unsqueeze_node.GetInputs();
  if (inputs.size() >= 2) {
    return ReadIntInitializerNoLimit(inputs[1]);
  }
  return GetIntsAttribute(unsqueeze_node, "axes");
}

bool IsZeroFloatConstantOfShape(Ort::ConstNode node) {
  if (!IsOnnxOp(node, "ConstantOfShape")) {
    return false;
  }

  Ort::ConstOpAttr attr;
  Ort::Status attr_status = node.GetAttributeByName("value", attr);
  if (!attr_status.IsOK()) {
    return true;
  }

  Ort::Value value{nullptr};
  Ort::Status status = attr.GetTensorAttributeAsOrtValue(value);
  if (!status.IsOK() || !value) {
    return false;
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      info.GetElementCount() != 1) {
    return false;
  }
  return value.GetTensorData<float>()[0] == 0.0f;
}

std::optional<std::vector<int64_t>> ConstantOfShapeOutputShape(
    Ort::ConstNode node, Ort::ConstValueInfo output_info) {
  auto output_shape = GetTensorShape(output_info);
  if (output_shape.has_value() && output_shape->size() == 2 &&
      (*output_shape)[1] > 0) {
    return output_shape;
  }

  std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  if (inputs.size() != 1) {
    return std::nullopt;
  }

  auto shape_from_initializer = ReadIntInitializerNoLimit(inputs[0]);
  if (shape_from_initializer.has_value() &&
      shape_from_initializer->size() == 2 && (*shape_from_initializer)[1] > 0) {
    return shape_from_initializer;
  }

  Ort::ValueInfoConsumerProducerInfo producer = inputs[0].GetProducerNode();
  if (!producer.node || !IsOnnxOp(producer.node, "Shape")) {
    return std::nullopt;
  }

  std::vector<Ort::ConstValueInfo> shape_inputs = producer.node.GetInputs();
  if (shape_inputs.size() != 1) {
    return std::nullopt;
  }
  auto shape = GetTensorShape(shape_inputs[0]);
  if (!shape.has_value() || shape->size() != 2 || (*shape)[1] <= 0) {
    return std::nullopt;
  }
  return shape;
}

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

bool CanFuseConcatSplit(
    Ort::ConstNode concat_node, Ort::ConstNode split_node,
    const std::unordered_set<std::string>& graph_output_names,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  if (concat_inputs.empty() || concat_outputs.size() != 1 ||
      split_inputs.size() < 1 || split_inputs.size() > 2 ||
      split_outputs.empty() ||
      graph_output_names.count(Name(concat_outputs[0])) != 0 ||
      Name(concat_outputs[0]) != Name(split_inputs[0])) {
    return false;
  }

  auto concat_axis_attr = GetIntAttribute(concat_node, "axis");
  auto split_axis_attr = GetIntAttribute(split_node, "axis");
  int64_t concat_axis = 0;
  int64_t split_axis = 0;
  if (!concat_axis_attr.has_value() ||
      !NormalizeAxis(*concat_axis_attr, 2, concat_axis) ||
      !NormalizeAxis(split_axis_attr.value_or(0), 2, split_axis) ||
      concat_axis != 1 || split_axis != 1) {
    return false;
  }

  if (split_inputs.size() != 2) {
    return false;
  }
  auto split_sizes = ReadIntInitializerNoLimit(split_inputs[1]);
  if (!split_sizes.has_value() || split_sizes->size() != split_outputs.size()) {
    return false;
  }

  std::vector<int64_t> concat_widths;
  concat_widths.reserve(concat_inputs.size());
  int64_t total_width = 0;
  for (Ort::ConstValueInfo input : concat_inputs) {
    if (!IsFloatTensorValueInfo(input)) {
      return false;
    }
    auto shape = GetTensorShape(input);
    if (!shape.has_value() || shape->size() != 2 || (*shape)[1] <= 0) {
      return false;
    }
    concat_widths.push_back((*shape)[1]);
    total_width += (*shape)[1];
  }

  int64_t split_total = 0;
  for (int64_t size : *split_sizes) {
    if (size <= 0) {
      return false;
    }
    split_total += size;
  }
  if (split_total != total_width) {
    return false;
  }

  std::unordered_set<std::string> split_output_names;
  std::unordered_map<std::string, int64_t> split_widths;
  for (size_t i = 0; i < split_outputs.size(); ++i) {
    split_output_names.insert(Name(split_outputs[i]));
    split_widths.emplace(Name(split_outputs[i]), (*split_sizes)[i]);
  }

  int64_t split_offset = 0;
  size_t concat_input_index = 0;
  int64_t concat_offset = 0;
  for (int64_t split_size : *split_sizes) {
    while (concat_input_index < concat_widths.size() &&
           split_offset >= concat_offset + concat_widths[concat_input_index]) {
      concat_offset += concat_widths[concat_input_index];
      ++concat_input_index;
    }
    if (concat_input_index >= concat_widths.size()) {
      return false;
    }
    const int64_t local_start = split_offset - concat_offset;
    if (local_start < 0 ||
        local_start + split_size > concat_widths[concat_input_index]) {
      return false;
    }
    split_offset += split_size;
  }

  fusion_nodes = {concat_node, split_node};
  for (Ort::ConstValueInfo split_output : split_outputs) {
    for (const auto& consumer : split_output.GetConsumers()) {
      Ort::ConstNode downstream_node = consumer.node;
      if (IsOnnxOp(downstream_node, "Concat")) {
        auto axis_attr = GetIntAttribute(downstream_node, "axis");
        int64_t downstream_axis = 0;
        if (!axis_attr.has_value() ||
            !NormalizeAxis(*axis_attr, 2, downstream_axis) ||
            downstream_axis != 1) {
          continue;
        }
        bool all_inputs_from_split = true;
        for (Ort::ConstValueInfo input : downstream_node.GetInputs()) {
          if (split_output_names.count(Name(input)) == 0) {
            all_inputs_from_split = false;
            break;
          }
        }
        if (all_inputs_from_split) {
          fusion_nodes.push_back(downstream_node);
        }
      } else if (IsOnnxOp(downstream_node, "Sum")) {
        std::vector<Ort::ConstValueInfo> sum_inputs =
            downstream_node.GetInputs();
        std::vector<Ort::ConstValueInfo> sum_outputs =
            downstream_node.GetOutputs();
        if (sum_inputs.size() < 2 || sum_outputs.size() != 1 ||
            !IsFloatTensorValueInfo(sum_outputs[0])) {
          continue;
        }
        int64_t sum_width = -1;
        bool can_fuse_sum = true;
        for (Ort::ConstValueInfo input : sum_inputs) {
          auto width_it = split_widths.find(Name(input));
          if (width_it == split_widths.end()) {
            can_fuse_sum = false;
            break;
          }
          if (sum_width < 0) {
            sum_width = width_it->second;
          } else if (sum_width != width_it->second) {
            can_fuse_sum = false;
            break;
          }
        }
        if (can_fuse_sum) {
          fusion_nodes.push_back(downstream_node);
        }
      }
    }
  }

  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  fusion_nodes.erase(std::unique(fusion_nodes.begin(), fusion_nodes.end(),
                                 [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
                                   return lhs.GetId() == rhs.GetId();
                                 }),
                     fusion_nodes.end());
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindConcatSplitFusions(
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
    if (concat_outputs.size() != 1) {
      continue;
    }
    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
        concat_outputs[0].GetConsumers();
    if (consumers.size() != 1) {
      continue;
    }

    Ort::ConstNode split_node = consumers[0].node;
    if (!IsOnnxOp(split_node, "Split") ||
        fused_node_ids.count(split_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseConcatSplit(concat_node, split_node, graph_output_names,
                            fusion_nodes)) {
      continue;
    }

    for (Ort::ConstNode node : fusion_nodes) {
      fused_node_ids.insert(node.GetId());
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

bool CanFuseSplitUnsqueezeConcat(
    Ort::ConstNode reshape_node, Ort::ConstNode split_node,
    const std::vector<Ort::ConstNode>& unsqueeze_nodes,
    Ort::ConstNode concat_node, Ort::ConstNode transpose_node,
    const std::unordered_set<std::string>& graph_output_names) {
  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (split_inputs.empty() || split_inputs.size() > 2 ||
      split_outputs.size() < 2 ||
      unsqueeze_nodes.size() != split_outputs.size() ||
      concat_inputs.size() != split_outputs.size() ||
      concat_outputs.size() != 1) {
    return false;
  }

  Ort::ConstValueInfo packed_value = split_inputs[0];
  Ort::ConstValueInfo source_value = split_inputs[0];
  std::optional<std::vector<int64_t>> reshape_target_shape;
  if (reshape_node) {
    std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
    std::vector<Ort::ConstValueInfo> reshape_outputs =
        reshape_node.GetOutputs();
    if (reshape_inputs.empty() || reshape_outputs.size() != 1 ||
        Name(split_inputs[0]) != Name(reshape_outputs[0]) ||
        graph_output_names.count(Name(reshape_outputs[0])) != 0) {
      return false;
    }
    packed_value = reshape_outputs[0];
    source_value = reshape_inputs[0];
    if (reshape_inputs.size() >= 2) {
      reshape_target_shape = ReadIntInitializerNoLimit(reshape_inputs[1]);
    }
  }

  if (!IsFloatTensorValueInfo(source_value) ||
      !IsFloatTensorValueInfo(packed_value) ||
      !IsFloatTensorValueInfo(concat_outputs[0])) {
    return false;
  }

  const int64_t part_count = static_cast<int64_t>(split_outputs.size());
  int64_t batch = -1;
  int64_t sequence = -1;
  int64_t packed_width = -1;
  int64_t part_width_from_outputs = -1;

  if (reshape_target_shape.has_value()) {
    if (reshape_target_shape->size() != 3) {
      return false;
    }
    if ((*reshape_target_shape)[0] > 0) {
      batch = (*reshape_target_shape)[0];
    }
    if ((*reshape_target_shape)[1] > 0) {
      sequence = (*reshape_target_shape)[1];
    }
    if ((*reshape_target_shape)[2] > 0) {
      packed_width = (*reshape_target_shape)[2];
    }
  }

  auto packed_shape = GetTensorShape(packed_value);
  if (packed_shape.has_value()) {
    if (packed_shape->size() != 3) {
      return false;
    }
    if ((*packed_shape)[0] > 0) {
      if (batch > 0 && batch != (*packed_shape)[0]) {
        return false;
      }
      batch = (*packed_shape)[0];
    }
    if ((*packed_shape)[1] > 0) {
      if (sequence > 0 && sequence != (*packed_shape)[1]) {
        return false;
      }
      sequence = (*packed_shape)[1];
    }
    if ((*packed_shape)[2] > 0) {
      if (packed_width > 0 && packed_width != (*packed_shape)[2]) {
        return false;
      }
      packed_width = (*packed_shape)[2];
    }
  }

  auto first_split_shape = GetTensorShape(split_outputs[0]);
  if (first_split_shape.has_value()) {
    if (first_split_shape->size() != 3) {
      return false;
    }
    if ((*first_split_shape)[0] > 0) {
      if (batch > 0 && batch != (*first_split_shape)[0]) {
        return false;
      }
      batch = (*first_split_shape)[0];
    }
    if ((*first_split_shape)[1] > 0) {
      if (sequence > 0 && sequence != (*first_split_shape)[1]) {
        return false;
      }
      sequence = (*first_split_shape)[1];
    }
    if ((*first_split_shape)[2] > 0) {
      part_width_from_outputs = (*first_split_shape)[2];
    }
  }

  auto split_axis_attr = GetIntAttribute(split_node, "axis");
  int64_t split_axis = 0;
  if (!NormalizeAxis(split_axis_attr.value_or(0), 3, split_axis) ||
      split_axis != 2) {
    return false;
  }

  auto concat_axis_attr = GetIntAttribute(concat_node, "axis");
  int64_t concat_axis = 0;
  if (!concat_axis_attr.has_value() ||
      !NormalizeAxis(*concat_axis_attr, 4, concat_axis) || concat_axis != 0) {
    return false;
  }

  std::vector<int64_t> split_sizes;
  int64_t part_width = part_width_from_outputs;
  if (split_inputs.size() == 2) {
    auto split_initializer = ReadIntInitializerNoLimit(split_inputs[1]);
    if (!split_initializer.has_value() ||
        split_initializer->size() != split_outputs.size()) {
      return false;
    }
    split_sizes = std::move(*split_initializer);
    if (split_sizes.empty() || split_sizes[0] <= 0) {
      return false;
    }
    part_width = split_sizes[0];
  } else if (packed_width > 0) {
    if (packed_width % part_count != 0) {
      return false;
    }
    split_sizes.assign(split_outputs.size(), packed_width / part_count);
    part_width = split_sizes[0];
  } else if (part_width_from_outputs > 0) {
    packed_width = part_width_from_outputs * part_count;
    split_sizes.assign(split_outputs.size(), part_width_from_outputs);
    part_width = part_width_from_outputs;
  }

  if (part_width_from_outputs > 0 && part_width > 0 &&
      part_width != part_width_from_outputs) {
    return false;
  }
  int64_t split_total = 0;
  for (int64_t split_size : split_sizes) {
    if (split_size <= 0 || (part_width > 0 && split_size != part_width)) {
      return false;
    }
    split_total += split_size;
  }
  if (part_width <= 0 && !split_sizes.empty()) {
    part_width = split_sizes[0];
  }
  if (packed_width <= 0 && split_total > 0) {
    packed_width = split_total;
  }
  if (packed_width > 0 && split_total > 0 && split_total != packed_width) {
    return false;
  }

  for (size_t i = 0; i < split_outputs.size(); ++i) {
    Ort::ConstValueInfo split_output = split_outputs[i];
    Ort::ConstNode unsqueeze_node = unsqueeze_nodes[i];
    std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
        unsqueeze_node.GetInputs();
    std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
        unsqueeze_node.GetOutputs();
    if (unsqueeze_inputs.empty() || unsqueeze_outputs.size() != 1 ||
        Name(unsqueeze_inputs[0]) != Name(split_output) ||
        Name(concat_inputs[i]) != Name(unsqueeze_outputs[0]) ||
        graph_output_names.count(Name(split_output)) != 0 ||
        graph_output_names.count(Name(unsqueeze_outputs[0])) != 0 ||
        !IsFloatTensorValueInfo(split_output) ||
        !IsFloatTensorValueInfo(unsqueeze_outputs[0])) {
      return false;
    }

    auto split_shape = GetTensorShape(split_output);
    if (split_shape.has_value()) {
      if (split_shape->size() != 3 ||
          ((*split_shape)[0] > 0 && batch > 0 && (*split_shape)[0] != batch) ||
          ((*split_shape)[1] > 0 && sequence > 0 &&
           (*split_shape)[1] != sequence) ||
          ((*split_shape)[2] > 0 && part_width > 0 &&
           (*split_shape)[2] != part_width)) {
        return false;
      }
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> split_consumers =
        split_output.GetConsumers();
    if (split_consumers.size() != 1 ||
        split_consumers[0].node.GetId() != unsqueeze_node.GetId() ||
        split_consumers[0].index != 0) {
      return false;
    }

    auto axes = ReadUnsqueezeAxes(unsqueeze_node);
    int64_t unsqueeze_axis = 0;
    if (!axes.has_value() || axes->size() != 1 ||
        !NormalizeAxis((*axes)[0], 4, unsqueeze_axis) || unsqueeze_axis != 0) {
      return false;
    }

    auto unsqueeze_shape = GetTensorShape(unsqueeze_outputs[0]);
    if (unsqueeze_shape.has_value()) {
      if (unsqueeze_shape->size() != 4 ||
          ((*unsqueeze_shape)[0] > 0 && (*unsqueeze_shape)[0] != 1) ||
          ((*unsqueeze_shape)[1] > 0 && batch > 0 &&
           (*unsqueeze_shape)[1] != batch) ||
          ((*unsqueeze_shape)[2] > 0 && sequence > 0 &&
           (*unsqueeze_shape)[2] != sequence) ||
          ((*unsqueeze_shape)[3] > 0 && part_width > 0 &&
           (*unsqueeze_shape)[3] != part_width)) {
        return false;
      }
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> unsqueeze_consumers =
        unsqueeze_outputs[0].GetConsumers();
    if (unsqueeze_consumers.size() != 1 ||
        unsqueeze_consumers[0].node.GetId() != concat_node.GetId() ||
        unsqueeze_consumers[0].index != static_cast<int64_t>(i)) {
      return false;
    }
  }

  auto concat_shape = GetTensorShape(concat_outputs[0]);
  if (concat_shape.has_value()) {
    if (concat_shape->size() != 4 ||
        ((*concat_shape)[0] > 0 && (*concat_shape)[0] != part_count) ||
        ((*concat_shape)[1] > 0 && batch > 0 && (*concat_shape)[1] != batch) ||
        ((*concat_shape)[2] > 0 && sequence > 0 &&
         (*concat_shape)[2] != sequence) ||
        ((*concat_shape)[3] > 0 && part_width > 0 &&
         (*concat_shape)[3] != part_width)) {
      return false;
    }
  }

  if (!transpose_node) {
    return true;
  }

  if (graph_output_names.count(Name(concat_outputs[0])) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> transpose_inputs =
      transpose_node.GetInputs();
  std::vector<Ort::ConstValueInfo> transpose_outputs =
      transpose_node.GetOutputs();
  if (transpose_inputs.size() != 1 || transpose_outputs.size() != 1 ||
      Name(transpose_inputs[0]) != Name(concat_outputs[0]) ||
      !IsFloatTensorValueInfo(transpose_outputs[0])) {
    return false;
  }

  auto perm = GetIntsAttribute(transpose_node, "perm");
  if (!perm.has_value() || *perm != std::vector<int64_t>({0, 1, 3, 2})) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> concat_consumers =
      concat_outputs[0].GetConsumers();
  if (concat_consumers.size() != 1 ||
      concat_consumers[0].node.GetId() != transpose_node.GetId() ||
      concat_consumers[0].index != 0) {
    return false;
  }

  auto transpose_shape = GetTensorShape(transpose_outputs[0]);
  if (transpose_shape.has_value()) {
    if (transpose_shape->size() != 4 ||
        ((*transpose_shape)[0] > 0 && (*transpose_shape)[0] != part_count) ||
        ((*transpose_shape)[1] > 0 && batch > 0 &&
         (*transpose_shape)[1] != batch) ||
        ((*transpose_shape)[2] > 0 && part_width > 0 &&
         (*transpose_shape)[2] != part_width) ||
        ((*transpose_shape)[3] > 0 && sequence > 0 &&
         (*transpose_shape)[3] != sequence)) {
      return false;
    }
  }

  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindSplitUnsqueezeConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode split_node : all_nodes) {
    if (!IsOnnxOp(split_node, "Split") ||
        fused_node_ids.count(split_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
    std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
    if (split_inputs.empty() || split_outputs.size() < 2) {
      continue;
    }

    Ort::ConstNode reshape_node{nullptr};
    Ort::ValueInfoConsumerProducerInfo producer =
        split_inputs[0].GetProducerNode();
    if (producer.node && IsOnnxOp(producer.node, "Reshape") &&
        fused_node_ids.count(producer.node.GetId()) == 0) {
      std::vector<Ort::ConstValueInfo> reshape_outputs =
          producer.node.GetOutputs();
      if (reshape_outputs.size() == 1 &&
          reshape_outputs[0].GetConsumers().size() == 1) {
        reshape_node = producer.node;
      }
    }

    std::vector<Ort::ConstNode> unsqueeze_nodes;
    unsqueeze_nodes.reserve(split_outputs.size());
    Ort::ConstNode concat_node{nullptr};
    bool all_outputs_feed_same_concat = true;
    for (Ort::ConstValueInfo split_output : split_outputs) {
      std::vector<Ort::ValueInfoConsumerProducerInfo> split_consumers =
          split_output.GetConsumers();
      if (split_consumers.size() != 1 ||
          !IsOnnxOp(split_consumers[0].node, "Unsqueeze") ||
          fused_node_ids.count(split_consumers[0].node.GetId()) != 0) {
        all_outputs_feed_same_concat = false;
        break;
      }

      Ort::ConstNode unsqueeze_node = split_consumers[0].node;
      std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
          unsqueeze_node.GetOutputs();
      if (unsqueeze_outputs.size() != 1) {
        all_outputs_feed_same_concat = false;
        break;
      }
      std::vector<Ort::ValueInfoConsumerProducerInfo> unsqueeze_consumers =
          unsqueeze_outputs[0].GetConsumers();
      if (unsqueeze_consumers.size() != 1 ||
          !IsOnnxOp(unsqueeze_consumers[0].node, "Concat")) {
        all_outputs_feed_same_concat = false;
        break;
      }
      if (!concat_node) {
        concat_node = unsqueeze_consumers[0].node;
      } else if (concat_node.GetId() != unsqueeze_consumers[0].node.GetId()) {
        all_outputs_feed_same_concat = false;
        break;
      }
      unsqueeze_nodes.push_back(unsqueeze_node);
    }
    if (!all_outputs_feed_same_concat || !concat_node ||
        fused_node_ids.count(concat_node.GetId()) != 0) {
      continue;
    }

    Ort::ConstNode transpose_node{nullptr};
    std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
    if (concat_outputs.size() == 1) {
      std::vector<Ort::ValueInfoConsumerProducerInfo> concat_consumers =
          concat_outputs[0].GetConsumers();
      if (concat_consumers.size() == 1 &&
          IsOnnxOp(concat_consumers[0].node, "Transpose") &&
          fused_node_ids.count(concat_consumers[0].node.GetId()) == 0) {
        transpose_node = concat_consumers[0].node;
      }
    }

    bool fuse_transpose =
        transpose_node && CanFuseSplitUnsqueezeConcat(
                              reshape_node, split_node, unsqueeze_nodes,
                              concat_node, transpose_node, graph_output_names);
    bool fuse_without_transpose = CanFuseSplitUnsqueezeConcat(
        reshape_node, split_node, unsqueeze_nodes, concat_node,
        Ort::ConstNode{nullptr}, graph_output_names);
    if (!fuse_transpose && !fuse_without_transpose) {
      continue;
    }

    std::vector<Ort::ConstNode> fusion_nodes;
    fusion_nodes.reserve(2 + (reshape_node ? 1 : 0) + unsqueeze_nodes.size() +
                         (fuse_transpose ? 1 : 0));
    if (reshape_node) {
      fusion_nodes.push_back(reshape_node);
    }
    fusion_nodes.push_back(split_node);
    fusion_nodes.insert(fusion_nodes.end(), unsqueeze_nodes.begin(),
                        unsqueeze_nodes.end());
    fusion_nodes.push_back(concat_node);
    if (fuse_transpose) {
      fusion_nodes.push_back(transpose_node);
    }

    for (Ort::ConstNode node : fusion_nodes) {
      fused_node_ids.insert(node.GetId());
    }
    fusions.push_back(std::move(fusion_nodes));
  }

  return fusions;
}

bool CanFuseSplitConcatReorder(
    Ort::ConstNode reshape_node, Ort::ConstNode split_node,
    Ort::ConstNode concat_node,
    const std::unordered_set<std::string>& graph_output_names) {
  std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reshape_outputs = reshape_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (reshape_inputs.empty() || reshape_outputs.size() != 1 ||
      split_inputs.empty() || split_inputs.size() > 2 ||
      split_outputs.size() < 2 ||
      concat_inputs.size() != split_outputs.size() ||
      concat_outputs.size() != 1 ||
      Name(split_inputs[0]) != Name(reshape_outputs[0])) {
    return false;
  }

  if (graph_output_names.count(Name(reshape_outputs[0])) != 0) {
    return false;
  }

  if (!IsFloatTensorValueInfo(reshape_inputs[0]) ||
      !IsFloatTensorValueInfo(reshape_outputs[0]) ||
      !IsFloatTensorValueInfo(concat_outputs[0])) {
    return false;
  }

  auto reshape_shape = GetTensorShape(reshape_outputs[0]);
  if (!reshape_shape.has_value() || reshape_shape->size() != 3 ||
      (*reshape_shape)[1] <= 0 || (*reshape_shape)[2] <= 0) {
    return false;
  }
  const int64_t batch = (*reshape_shape)[0];
  const int64_t sequence = (*reshape_shape)[1];
  const int64_t packed_width = (*reshape_shape)[2];

  auto split_axis_attr = GetIntAttribute(split_node, "axis");
  int64_t split_axis = 0;
  if (!NormalizeAxis(split_axis_attr.value_or(0), 3, split_axis) ||
      split_axis != 2) {
    return false;
  }

  auto concat_axis_attr = GetIntAttribute(concat_node, "axis");
  int64_t concat_axis = 0;
  if (!concat_axis_attr.has_value() ||
      !NormalizeAxis(*concat_axis_attr, 3, concat_axis) || concat_axis != 0) {
    return false;
  }

  std::vector<int64_t> split_sizes;
  if (split_inputs.size() == 2) {
    auto split_initializer = ReadIntInitializerNoLimit(split_inputs[1]);
    if (!split_initializer.has_value() ||
        split_initializer->size() != split_outputs.size()) {
      return false;
    }
    split_sizes = std::move(*split_initializer);
  } else {
    if (packed_width % static_cast<int64_t>(split_outputs.size()) != 0) {
      return false;
    }
    split_sizes.assign(
        split_outputs.size(),
        packed_width / static_cast<int64_t>(split_outputs.size()));
  }

  if (split_sizes.empty() || split_sizes[0] <= 0) {
    return false;
  }
  const int64_t part_width = split_sizes[0];
  int64_t split_total = 0;
  for (int64_t split_size : split_sizes) {
    if (split_size != part_width) {
      return false;
    }
    split_total += split_size;
  }
  if (split_total != packed_width) {
    return false;
  }

  for (size_t i = 0; i < split_outputs.size(); ++i) {
    Ort::ConstValueInfo split_output = split_outputs[i];
    if (Name(concat_inputs[i]) != Name(split_output) ||
        graph_output_names.count(Name(split_output)) != 0 ||
        !IsFloatTensorValueInfo(split_output)) {
      return false;
    }

    auto split_shape = GetTensorShape(split_output);
    if (!split_shape.has_value() || split_shape->size() != 3 ||
        ((*split_shape)[0] > 0 && batch > 0 && (*split_shape)[0] != batch) ||
        (*split_shape)[1] != sequence || (*split_shape)[2] != part_width) {
      return false;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
        split_output.GetConsumers();
    if (consumers.size() != 1 ||
        consumers[0].node.GetId() != concat_node.GetId() ||
        consumers[0].index != static_cast<int64_t>(i)) {
      return false;
    }
  }

  auto concat_shape = GetTensorShape(concat_outputs[0]);
  if (!concat_shape.has_value() || concat_shape->size() != 3 ||
      ((*concat_shape)[0] > 0 && batch > 0 &&
       (*concat_shape)[0] !=
           batch * static_cast<int64_t>(split_outputs.size())) ||
      (*concat_shape)[1] != sequence || (*concat_shape)[2] != part_width) {
    return false;
  }

  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindSplitConcatReorderFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode split_node : all_nodes) {
    if (!IsOnnxOp(split_node, "Split") ||
        fused_node_ids.count(split_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
    std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
    if (split_inputs.empty() || split_outputs.size() < 2) {
      continue;
    }

    Ort::ValueInfoConsumerProducerInfo producer =
        split_inputs[0].GetProducerNode();
    if (!producer.node || !IsOnnxOp(producer.node, "Reshape") ||
        fused_node_ids.count(producer.node.GetId()) != 0) {
      continue;
    }
    Ort::ConstNode reshape_node = producer.node;

    std::vector<Ort::ConstValueInfo> reshape_outputs =
        reshape_node.GetOutputs();
    if (reshape_outputs.size() != 1 ||
        reshape_outputs[0].GetConsumers().size() != 1) {
      continue;
    }

    Ort::ConstNode concat_node{nullptr};
    bool all_outputs_feed_same_concat = true;
    for (Ort::ConstValueInfo split_output : split_outputs) {
      std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
          split_output.GetConsumers();
      if (consumers.size() != 1 || !IsOnnxOp(consumers[0].node, "Concat")) {
        all_outputs_feed_same_concat = false;
        break;
      }
      if (!concat_node) {
        concat_node = consumers[0].node;
      } else if (concat_node.GetId() != consumers[0].node.GetId()) {
        all_outputs_feed_same_concat = false;
        break;
      }
    }
    if (!all_outputs_feed_same_concat || !concat_node ||
        fused_node_ids.count(concat_node.GetId()) != 0) {
      continue;
    }

    if (!CanFuseSplitConcatReorder(reshape_node, split_node, concat_node,
                                   graph_output_names)) {
      continue;
    }

    fusions.push_back({reshape_node, split_node, concat_node});
    fused_node_ids.insert(reshape_node.GetId());
    fused_node_ids.insert(split_node.GetId());
    fused_node_ids.insert(concat_node.GetId());
  }

  return fusions;
}

bool CanFuseSliceConcat(
    Ort::ConstNode concat_node,
    const std::unordered_set<std::string>& graph_output_names,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_inputs.size() < 8 || concat_outputs.size() != 1 ||
      graph_output_names.count(Name(concat_outputs[0])) != 0) {
    return false;
  }

  auto axis_attr = GetIntAttribute(concat_node, "axis");
  int64_t axis = 0;
  if (!axis_attr.has_value() || !NormalizeAxis(*axis_attr, 2, axis) ||
      axis != 1) {
    return false;
  }

  fusion_nodes.clear();
  fusion_nodes.reserve(concat_inputs.size() + 1);
  std::unordered_set<size_t> seen_slice_node_ids;
  bool has_fused_slice_input = false;
  for (Ort::ConstValueInfo concat_input : concat_inputs) {
    Ort::ValueInfoConsumerProducerInfo producer =
        concat_input.GetProducerNode();
    if (!producer.node) {
      auto shape = GetTensorShape(concat_input);
      if (!IsFloatTensorValueInfo(concat_input) || !shape.has_value() ||
          shape->size() != 2 || (*shape)[1] <= 0) {
        return false;
      }
      continue;
    }

    Ort::ConstNode slice_node = producer.node;
    const bool is_slice_input = IsOnnxOp(slice_node, "Slice");
    const bool is_zero_constant_input = IsOnnxOp(slice_node, "ConstantOfShape");
    if (!is_slice_input && !is_zero_constant_input) {
      auto shape = GetTensorShape(concat_input);
      if (!IsFloatTensorValueInfo(concat_input) || !shape.has_value() ||
          shape->size() != 2 || (*shape)[1] <= 0) {
        return false;
      }
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
        concat_input.GetConsumers();
    if (consumers.empty()) {
      return false;
    }
    for (const auto& consumer : consumers) {
      if (consumer.node.GetId() != concat_node.GetId()) {
        return false;
      }
    }

    has_fused_slice_input = true;
    if (IsOnnxOp(slice_node, "ConstantOfShape")) {
      if (!IsZeroFloatConstantOfShape(slice_node)) {
        return false;
      }
      auto output_shape = ConstantOfShapeOutputShape(slice_node, concat_input);
      if (!output_shape.has_value()) {
        return false;
      }
      std::vector<Ort::ConstValueInfo> constant_inputs = slice_node.GetInputs();
      if (constant_inputs.size() == 1) {
        Ort::ValueInfoConsumerProducerInfo shape_producer =
            constant_inputs[0].GetProducerNode();
        if (shape_producer.node && IsOnnxOp(shape_producer.node, "Shape")) {
          std::vector<Ort::ValueInfoConsumerProducerInfo> shape_consumers =
              constant_inputs[0].GetConsumers();
          if (shape_consumers.size() != 1 ||
              shape_consumers[0].node.GetId() != slice_node.GetId()) {
            return false;
          }
          if (seen_slice_node_ids.insert(shape_producer.node.GetId()).second) {
            fusion_nodes.push_back(shape_producer.node);
          }
        }
      }
      if (seen_slice_node_ids.insert(slice_node.GetId()).second) {
        fusion_nodes.push_back(slice_node);
      }
      continue;
    }

    std::vector<Ort::ConstValueInfo> slice_inputs = slice_node.GetInputs();
    if (slice_inputs.size() < 3 || slice_inputs.size() > 5) {
      return false;
    }

    auto input_shape = GetTensorShape(slice_inputs[0]);
    if (input_shape.has_value() && input_shape->size() != 2) {
      return false;
    }

    auto starts = ReadIntInitializerNoLimit(slice_inputs[1]);
    auto ends = ReadIntInitializerNoLimit(slice_inputs[2]);
    if (!starts.has_value() || !ends.has_value() ||
        starts->size() != ends->size()) {
      return false;
    }
    std::vector<int64_t> axes(starts->size());
    std::iota(axes.begin(), axes.end(), 0);
    if (slice_inputs.size() > 3 && slice_inputs[3]) {
      auto axes_init = ReadIntInitializerNoLimit(slice_inputs[3]);
      if (!axes_init.has_value()) {
        return false;
      }
      axes = *axes_init;
    }
    std::vector<int64_t> steps(starts->size(), 1);
    if (slice_inputs.size() > 4 && slice_inputs[4]) {
      auto steps_init = ReadIntInitializerNoLimit(slice_inputs[4]);
      if (!steps_init.has_value()) {
        return false;
      }
      steps = *steps_init;
    }
    if (axes.size() != starts->size() || steps.size() != starts->size()) {
      return false;
    }

    bool has_col_slice = false;
    for (size_t i = 0; i < axes.size(); ++i) {
      int64_t slice_axis = axes[i] < 0 ? axes[i] + 2 : axes[i];
      if (slice_axis < 0 || slice_axis > 1 || steps[i] != 1) {
        return false;
      }
      if (slice_axis == 0) {
        if ((*starts)[i] != 0 ||
            (*ends)[i] < std::numeric_limits<int64_t>::max() / 4) {
          return false;
        }
      } else {
        if (((*starts)[i] < 0 || (*ends)[i] < 0) &&
            (!input_shape.has_value() || (*input_shape)[1] <= 0)) {
          return false;
        }
        int64_t start =
            (*starts)[i] < 0 ? (*starts)[i] + (*input_shape)[1] : (*starts)[i];
        int64_t end =
            (*ends)[i] < 0 ? (*ends)[i] + (*input_shape)[1] : (*ends)[i];
        if (input_shape.has_value() && (*input_shape)[1] > 0) {
          start = std::max<int64_t>(0, std::min(start, (*input_shape)[1]));
          end = std::max<int64_t>(0, std::min(end, (*input_shape)[1]));
        }
        if (end <= start) {
          return false;
        }
        has_col_slice = true;
      }
    }
    if (!has_col_slice) {
      return false;
    }

    if (seen_slice_node_ids.insert(slice_node.GetId()).second) {
      fusion_nodes.push_back(slice_node);
    }
  }

  fusion_nodes.push_back(concat_node);
  return has_fused_slice_input;
}

std::vector<std::vector<Ort::ConstNode>> FindSliceConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode concat_node : all_nodes) {
    if (!IsOnnxOp(concat_node, "Concat") ||
        fused_node_ids.count(concat_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseSliceConcat(concat_node, graph_output_names, fusion_nodes)) {
      continue;
    }

    bool overlaps_existing_fusion = false;
    for (Ort::ConstNode node : fusion_nodes) {
      if (fused_node_ids.count(node.GetId()) != 0) {
        overlaps_existing_fusion = true;
        break;
      }
    }
    if (overlaps_existing_fusion) {
      continue;
    }

    for (Ort::ConstNode node : fusion_nodes) {
      fused_node_ids.insert(node.GetId());
    }
    fusions.push_back(std::move(fusion_nodes));
  }

  return fusions;
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

bool CanFuseShapeReshapeFromGather(
    Ort::ConstNode gather_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::unordered_set<size_t>& group_node_ids) {
  if (!IsOnnxOp(gather_node, "Gather") ||
      fused_node_ids.count(gather_node.GetId()) != 0 ||
      GetIntAttribute(gather_node, "axis").value_or(0) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gather_outputs = gather_node.GetOutputs();
  if (gather_inputs.size() != 2 || gather_outputs.size() != 1 ||
      graph_output_names.count(Name(gather_outputs[0])) != 0 ||
      !IsSmallIntegerInitializer(gather_inputs[1])) {
    return false;
  }

  Ort::ConstNode cast_node = FindProducer(producers, gather_inputs[0]);
  if (!cast_node || !IsOnnxOp(cast_node, "Cast") ||
      fused_node_ids.count(cast_node.GetId()) != 0) {
    return false;
  }
  auto cast_to = GetIntAttribute(cast_node, "to");
  if (!cast_to.has_value() ||
      (*cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
       *cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> cast_inputs = cast_node.GetInputs();
  std::vector<Ort::ConstValueInfo> cast_outputs = cast_node.GetOutputs();
  if (cast_inputs.size() != 1 || cast_outputs.size() != 1 ||
      graph_output_names.count(Name(cast_outputs[0])) != 0 ||
      !HasOnlyConsumer(cast_outputs[0], gather_node, 0)) {
    return false;
  }

  Ort::ConstNode shape_input_producer = FindProducer(producers, cast_inputs[0]);
  if (!shape_input_producer) {
    return false;
  }

  Ort::ConstNode shape_node{nullptr};
  Ort::ConstNode pre_gather_node{nullptr};
  bool include_shape_node = false;
  if (IsOnnxOp(shape_input_producer, "Shape")) {
    shape_node = shape_input_producer;
    if (fused_node_ids.count(shape_node.GetId()) != 0) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
    std::vector<Ort::ConstValueInfo> shape_outputs = shape_node.GetOutputs();
    if (shape_inputs.size() != 1 || shape_outputs.size() != 1 ||
        graph_output_names.count(Name(shape_outputs[0])) != 0) {
      return false;
    }
    include_shape_node = HasOnlyConsumer(shape_outputs[0], cast_node, 0);
  } else if (IsOnnxOp(shape_input_producer, "Gather")) {
    pre_gather_node = shape_input_producer;
    if (fused_node_ids.count(pre_gather_node.GetId()) != 0 ||
        GetIntAttribute(pre_gather_node, "axis").value_or(0) != 0) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> pre_gather_inputs =
        pre_gather_node.GetInputs();
    std::vector<Ort::ConstValueInfo> pre_gather_outputs =
        pre_gather_node.GetOutputs();
    if (pre_gather_inputs.size() != 2 || pre_gather_outputs.size() != 1 ||
        graph_output_names.count(Name(pre_gather_outputs[0])) != 0 ||
        !IsSmallIntegerInitializer(pre_gather_inputs[1]) ||
        !HasOnlyConsumer(pre_gather_outputs[0], cast_node, 0)) {
      return false;
    }

    shape_node = FindProducer(producers, pre_gather_inputs[0]);
    if (!shape_node || !IsOnnxOp(shape_node, "Shape") ||
        fused_node_ids.count(shape_node.GetId()) != 0) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
    std::vector<Ort::ConstValueInfo> shape_outputs = shape_node.GetOutputs();
    if (shape_inputs.size() != 1 || shape_outputs.size() != 1 ||
        graph_output_names.count(Name(shape_outputs[0])) != 0) {
      return false;
    }
    include_shape_node = HasOnlyConsumer(shape_outputs[0], pre_gather_node, 0);
  } else {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> gather_consumers =
      gather_outputs[0].GetConsumers();
  if (gather_consumers.empty()) {
    return false;
  }

  group_node_ids.clear();
  if (include_shape_node) {
    group_node_ids.insert(shape_node.GetId());
  }
  if (pre_gather_node) {
    group_node_ids.insert(pre_gather_node.GetId());
  }
  group_node_ids.insert(cast_node.GetId());
  group_node_ids.insert(gather_node.GetId());

  for (const auto& gather_consumer : gather_consumers) {
    Ort::ConstNode concat_node = gather_consumer.node;
    if (!IsOnnxOp(concat_node, "Concat") ||
        fused_node_ids.count(concat_node.GetId()) != 0 ||
        GetIntAttribute(concat_node, "axis").value_or(0) != 0) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
    std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
    if (concat_inputs.size() < 2 || concat_outputs.size() != 1 ||
        graph_output_names.count(Name(concat_outputs[0])) != 0) {
      return false;
    }

    int gather_input_count = 0;
    for (Ort::ConstValueInfo concat_input : concat_inputs) {
      if (Name(concat_input) == Name(gather_outputs[0])) {
        ++gather_input_count;
      } else if (!IsSmallIntegerInitializer(concat_input)) {
        return false;
      }
    }
    if (gather_input_count != 1) {
      return false;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> concat_consumers =
        concat_outputs[0].GetConsumers();
    if (concat_consumers.size() != 1 || concat_consumers[0].index != 0) {
      return false;
    }

    Ort::ConstNode final_cast_node = concat_consumers[0].node;
    if (!IsOnnxOp(final_cast_node, "Cast") ||
        fused_node_ids.count(final_cast_node.GetId()) != 0 ||
        GetIntAttribute(final_cast_node, "to").value_or(-1) !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> final_cast_outputs =
        final_cast_node.GetOutputs();
    if (final_cast_outputs.size() != 1 ||
        graph_output_names.count(Name(final_cast_outputs[0])) != 0) {
      return false;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> reshape_consumers =
        final_cast_outputs[0].GetConsumers();
    if (reshape_consumers.empty()) {
      return false;
    }

    group_node_ids.insert(concat_node.GetId());
    group_node_ids.insert(final_cast_node.GetId());

    for (const auto& reshape_consumer : reshape_consumers) {
      Ort::ConstNode reshape_node = reshape_consumer.node;
      if (reshape_consumer.index != 1 || !IsOnnxOp(reshape_node, "Reshape") ||
          fused_node_ids.count(reshape_node.GetId()) != 0) {
        return false;
      }

      std::vector<Ort::ConstValueInfo> reshape_inputs =
          reshape_node.GetInputs();
      std::vector<Ort::ConstValueInfo> reshape_outputs =
          reshape_node.GetOutputs();
      if (reshape_inputs.size() != 2 || reshape_outputs.size() != 1 ||
          reshape_inputs[0].IsConstantInitializer()) {
        return false;
      }
      group_node_ids.insert(reshape_node.GetId());
    }
  }

  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindShapeReshapeFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);

  for (Ort::ConstNode node : all_nodes) {
    std::unordered_set<size_t> group_node_ids;
    if (!CanFuseShapeReshapeFromGather(node, producers, graph_output_names,
                                       fused_node_ids, group_node_ids)) {
      continue;
    }

    std::vector<Ort::ConstNode> fusion_nodes;
    for (Ort::ConstNode ordered_node : all_nodes) {
      if (group_node_ids.count(ordered_node.GetId()) != 0) {
        fusion_nodes.push_back(ordered_node);
      }
    }
    if (!fusion_nodes.empty()) {
      fusions.push_back(std::move(fusion_nodes));
      fused_node_ids.insert(group_node_ids.begin(), group_node_ids.end());
    }
  }

  return fusions;
}

bool AddFusionNode(Ort::ConstNode node,
                   const std::unordered_set<size_t>& fused_node_ids,
                   std::unordered_set<size_t>& selected_node_ids,
                   std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!node || fused_node_ids.count(node.GetId()) != 0) {
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

bool CanFuseTileConcat(
    Ort::ConstNode concat_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(concat_node, "Concat") ||
      fused_node_ids.count(concat_node.GetId()) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_inputs.empty() || concat_outputs.size() != 1 ||
      graph_output_names.count(Name(concat_outputs[0])) != 0) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  bool found_tile_input = false;
  for (size_t input_idx = 0; input_idx < concat_inputs.size(); ++input_idx) {
    auto producer_it = producers.find(Name(concat_inputs[input_idx]));
    if (producer_it == producers.end() ||
        !IsOnnxOp(producer_it->second, "Tile")) {
      continue;
    }

    Ort::ConstNode tile_node = producer_it->second;
    if (fused_node_ids.count(tile_node.GetId()) != 0) {
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

    found_tile_input = true;
    if (!AddFusionNode(tile_node, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }

  if (!found_tile_input) {
    return false;
  }
  if (!AddFusionNode(concat_node, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                     selected_node_ids);
}

std::vector<std::vector<Ort::ConstNode>> FindTileConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);
  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseTileConcat(node, producers, graph_output_names, fused_node_ids,
                           fusion_nodes)) {
      continue;
    }
    for (Ort::ConstNode fusion_node : fusion_nodes) {
      fused_node_ids.insert(fusion_node.GetId());
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
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

bool CanFuseCenteredReduce(
    Ort::ConstNode first_reduce_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsReduceSumOrProd(first_reduce_node) ||
      fused_node_ids.count(first_reduce_node.GetId()) != 0 ||
      GetIntAttribute(first_reduce_node, "keepdims").value_or(1) != 1) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> first_reduce_inputs =
      first_reduce_node.GetInputs();
  std::vector<Ort::ConstValueInfo> first_reduce_outputs =
      first_reduce_node.GetOutputs();
  if (first_reduce_inputs.size() < 2 || first_reduce_outputs.size() != 1 ||
      !IsFloatTensorValueInfo(first_reduce_inputs[0]) ||
      !IsFloatTensorValueInfo(first_reduce_outputs[0])) {
    return false;
  }

  auto input_shape = GetTensorShape(first_reduce_inputs[0]);
  if (!input_shape.has_value() || input_shape->size() < 2 ||
      input_shape->back() <= 0 ||
      !ReduceAxesAreLastDim(first_reduce_node, input_shape->size()) ||
      !ReduceOutputKeepsLastDim(first_reduce_outputs[0], *input_shape)) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> first_consumers =
      first_reduce_outputs[0].GetConsumers();
  Ort::ConstNode sub_node{nullptr};
  for (const auto& consumer : first_consumers) {
    if (consumer.index == 1 && IsOnnxOp(consumer.node, "Sub")) {
      sub_node = consumer.node;
      break;
    }
  }
  if (!sub_node || fused_node_ids.count(sub_node.GetId()) != 0 ||
      !ValueHasExternalConsumerOrGraphOutput(first_reduce_outputs[0], sub_node,
                                             graph_output_names)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> sub_inputs = sub_node.GetInputs();
  std::vector<Ort::ConstValueInfo> sub_outputs = sub_node.GetOutputs();
  if (sub_inputs.size() != 2 || sub_outputs.size() != 1 ||
      graph_output_names.count(Name(sub_outputs[0])) != 0 ||
      Name(sub_inputs[0]) != Name(first_reduce_inputs[0]) ||
      Name(sub_inputs[1]) != Name(first_reduce_outputs[0]) ||
      !IsFloatTensorValueInfo(sub_outputs[0])) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> sub_consumers =
      sub_outputs[0].GetConsumers();
  Ort::ConstNode mul_node{nullptr};
  for (const auto& consumer : sub_consumers) {
    if (IsOnnxOp(consumer.node, "Mul")) {
      mul_node = consumer.node;
      break;
    }
  }
  if (!mul_node || fused_node_ids.count(mul_node.GetId()) != 0 ||
      !ValueHasOnlyConsumers(sub_outputs[0], mul_node)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> mul_inputs = mul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> mul_outputs = mul_node.GetOutputs();
  if (mul_inputs.size() != 2 || mul_outputs.size() != 1 ||
      graph_output_names.count(Name(mul_outputs[0])) != 0 ||
      Name(mul_inputs[0]) != Name(sub_outputs[0]) ||
      Name(mul_inputs[1]) != Name(sub_outputs[0]) ||
      !IsFloatTensorValueInfo(mul_outputs[0])) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> mul_consumers =
      mul_outputs[0].GetConsumers();
  Ort::ConstNode second_reduce_node{nullptr};
  for (const auto& consumer : mul_consumers) {
    if (consumer.index == 0 && IsReduceSumOrProd(consumer.node)) {
      second_reduce_node = consumer.node;
      break;
    }
  }
  if (!second_reduce_node ||
      fused_node_ids.count(second_reduce_node.GetId()) != 0 ||
      !ValueHasOnlyConsumers(mul_outputs[0], second_reduce_node) ||
      GetIntAttribute(second_reduce_node, "keepdims").value_or(1) != 1) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> second_reduce_inputs =
      second_reduce_node.GetInputs();
  std::vector<Ort::ConstValueInfo> second_reduce_outputs =
      second_reduce_node.GetOutputs();
  if (second_reduce_inputs.size() < 2 || second_reduce_outputs.size() != 1 ||
      !ReduceAxesAreLastDim(second_reduce_node, input_shape->size()) ||
      !IsFloatTensorValueInfo(second_reduce_outputs[0]) ||
      !ReduceOutputKeepsLastDim(second_reduce_outputs[0], *input_shape)) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  for (Ort::ConstNode node :
       {first_reduce_node, sub_node, mul_node, second_reduce_node}) {
    if (!AddFusionNode(node, fused_node_ids, selected_node_ids, fusion_nodes)) {
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

std::vector<std::vector<Ort::ConstNode>> FindCenteredReduceFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode first_reduce_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseCenteredReduce(first_reduce_node, graph_output_names,
                               fused_node_ids, fusion_nodes)) {
      continue;
    }
    for (Ort::ConstNode node : fusion_nodes) {
      fused_node_ids.insert(node.GetId());
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

bool CanFuseSplitReduce(
    Ort::ConstNode split_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(split_node, "Split") ||
      fused_node_ids.count(split_node.GetId()) != 0 ||
      GetIntAttribute(split_node, "axis").value_or(0) != 1) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  if (split_inputs.size() != 2 || split_outputs.size() != 2 ||
      !IsFloatTensorValueInfo(split_inputs[0]) ||
      graph_output_names.count(Name(split_outputs[0])) != 0 ||
      graph_output_names.count(Name(split_outputs[1])) != 0) {
    return false;
  }

  auto input_shape = GetTensorShape(split_inputs[0]);
  if (!input_shape.has_value() || input_shape->size() != 3 ||
      (*input_shape)[1] <= 0 || (*input_shape)[2] <= 0) {
    return false;
  }

  auto split_sizes = ReadSmallIntInitializer(split_inputs[1]);
  if (!split_sizes.has_value() || split_sizes->size() != 2 ||
      (*split_sizes)[0] <= 0 || (*split_sizes)[1] <= 0 ||
      (*split_sizes)[0] + (*split_sizes)[1] != (*input_shape)[1]) {
    return false;
  }

  fusion_nodes.clear();
  fusion_nodes.push_back(split_node);
  for (Ort::ConstValueInfo split_output : split_outputs) {
    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
        split_output.GetConsumers();
    if (consumers.size() != 1 || consumers[0].index != 0) {
      return false;
    }

    Ort::ConstNode reduce_node = consumers[0].node;
    if (fused_node_ids.count(reduce_node.GetId()) != 0 ||
        !(IsOnnxOp(reduce_node, "ReduceProd") ||
          IsOnnxOp(reduce_node, "ReduceMean")) ||
        GetIntAttribute(reduce_node, "keepdims").value_or(1) != 0 ||
        !ReduceAxesInputIsAxis1(reduce_node)) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> reduce_outputs = reduce_node.GetOutputs();
    if (reduce_outputs.size() != 1 ||
        !IsFloatTensorValueInfo(reduce_outputs[0])) {
      return false;
    }
    fusion_nodes.push_back(reduce_node);
  }
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindSplitReduceFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode split_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseSplitReduce(split_node, graph_output_names, fused_node_ids,
                            fusion_nodes)) {
      continue;
    }
    for (Ort::ConstNode node : fusion_nodes) {
      fused_node_ids.insert(node.GetId());
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

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
    if (add_consumers.size() == 1 && add_consumers[0].index == 0) {
      Ort::ConstNode activation_node = add_consumers[0].node;
      if (IsLinearActivationNode(activation_node) &&
          fused_node_ids.count(activation_node.GetId()) == 0 &&
          CanFuseMatMulAddActivation(matmul_node, add_node, activation_node,
                                     matmul_consumers[0].index)) {
        fusions.push_back({matmul_node, add_node, activation_node});
        fused_node_ids.insert(matmul_node.GetId());
        fused_node_ids.insert(add_node.GetId());
        fused_node_ids.insert(activation_node.GetId());
        continue;
      }
    }

    if (!CanFuseMatMulAdd(matmul_node, add_node, matmul_consumers[0].index)) {
      continue;
    }

    fusions.push_back({matmul_node, add_node});
    fused_node_ids.insert(matmul_node.GetId());
    fused_node_ids.insert(add_node.GetId());
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
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

  IGNORE_ORTSTATUS(ort_api_.Logger_LogMessage(
      &logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
      ("MUSAExecutionProvider has been created with name " + name_).c_str(),
      ORT_FILE, __LINE__, __FUNCTION__));
}

MusaEp::~MusaEp() = default;

/*static*/
OrtStatus* ORT_API_CALL MusaEp::CreateSyncStreamForDeviceImpl(
    OrtEp* this_ptr, const OrtMemoryDevice* memory_device,
    OrtSyncStreamImpl** stream) noexcept {
  auto& ep = *static_cast<MusaEp*>(this_ptr);
  *stream = nullptr;

  if (ep.ep_api_.MemoryDevice_GetDeviceType(memory_device) !=
          OrtMemoryInfoDeviceType_GPU ||
      ep.ep_api_.MemoryDevice_GetVendorId(memory_device) !=
          ep.factory_.VendorId() ||
      ep.ep_api_.MemoryDevice_GetMemoryType(memory_device) !=
          OrtDeviceMemoryType_DEFAULT) {
    return nullptr;
  }

  const MusaProviderOptions& options = ep.config_.provider_options;

  try {
    if (options.has_user_compute_stream != 0) {
      auto sync_stream = std::make_unique<MusaSyncStream>(
          ep.ort_api_, options.user_compute_stream);
      *stream = sync_stream.release();
      return nullptr;
    }

    auto sync_stream = std::make_unique<MusaSyncStream>(ep.ort_api_);
    *stream = sync_stream.release();
    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return ep.ort_api_.CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

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

    RETURN_IF_ERROR(DumpGraphToMermaidIfEnabled(*ort_graph));

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
    std::vector<std::vector<Ort::ConstNode>> split_unsqueeze_concat_fusions =
        FindSplitUnsqueezeConcatFusions(all_nodes, graph_output_names,
                                        fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> split_concat_reorder_fusions =
        FindSplitConcatReorderFusions(all_nodes, graph_output_names,
                                      fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> concat_matmul_fusions =
        FindConcatMatMulFusions(all_nodes, graph_output_names, fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> concat_split_fusions =
        FindConcatSplitFusions(all_nodes, graph_output_names, fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> slice_concat_fusions =
        FindSliceConcatFusions(all_nodes, graph_output_names, fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> tile_concat_fusions =
        FindTileConcatFusions(all_nodes, graph_output_names, fused_node_ids);

    for (const auto& fusion_nodes : split_unsqueeze_concat_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }
    for (const auto& fusion_nodes : split_concat_reorder_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }
    for (const auto& fusion_nodes : concat_matmul_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }
    for (const auto& fusion_nodes : concat_split_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }
    for (const auto& fusion_nodes : slice_concat_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }
    for (const auto& fusion_nodes : tile_concat_fusions) {
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
    std::vector<std::vector<Ort::ConstNode>> shape_reshape_fusions =
        FindShapeReshapeFusions(all_nodes, graph_output_names, fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> centered_reduce_fusions =
        FindCenteredReduceFusions(all_nodes, graph_output_names,
                                  fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> split_reduce_fusions =
        FindSplitReduceFusions(all_nodes, graph_output_names, fused_node_ids);

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

    for (const auto& fusion_nodes : shape_reshape_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = true;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    for (const auto& fusion_nodes : centered_reduce_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    for (const auto& fusion_nodes : split_reduce_fusions) {
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
