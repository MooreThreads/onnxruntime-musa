// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
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
#include <vector>

#include "ep_factory.h"
#include "ep_profiling.h"
#include "fusion/centered_reduce_fusion.h"
#include "fusion/concat_matmul_fusion.h"
#include "fusion/concat_split_fusion.h"
#include "fusion/fusion_node_compute.h"
#include "fusion/linear_fusion.h"
#include "fusion/masked_gather_reduce_fusion.h"
#include "fusion/pow_affine_split_reduce_fusion.h"
#include "fusion/shape_expand_fusion.h"
#include "fusion/shape_cast_concat_fusion.h"
#include "fusion/shape_cast_reshape_fusion.h"
#include "fusion/shape_cast_split_fusion.h"
#include "fusion/shape_cast_source_fusion.h"
#include "fusion/shape_cast_transpose_fusion.h"
#include "fusion/shape_gather_fusion.h"
#include "fusion/shape_reshape_fusion.h"
#include "fusion/slice_concat_fusion.h"
#include "fusion/slice_sum_concat_fusion.h"
#include "fusion/split_reduce_fusion.h"
#include "fusion/tile_mask_select_fusion.h"
#include "kernels/tensor/slice_sum_concat_impl.h"
#include "plugin_ep_utils.h"

namespace {

constexpr int64_t kMaxCpuMetadataElements = 16;

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(const Ort::ConstNode& node, const char* op_type) {
  return node && node.GetOperatorType() == op_type &&
         IsOnnxDomain(node.GetDomain());
}

std::vector<Ort::ConstNode> GetOutputNodes(
    const std::vector<Ort::ConstValueInfo>& node_outputs) {
  std::vector<Ort::ConstNode> output_nodes;
  for (Ort::ConstValueInfo output : node_outputs) {
    if (!output) {
      continue;
    }
    for (const auto& consumer : output.GetConsumers()) {
      output_nodes.push_back(consumer.node);
    }
  }
  return output_nodes;
}

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
}

void DebugPrintFusions(const char* label,
                       const std::vector<std::vector<Ort::ConstNode>>& fusions) {
  if (std::getenv("MUSA_EP_DEBUG_FUSIONS") == nullptr) {
    return;
  }
  for (const auto& fusion_nodes : fusions) {
    std::cerr << "[musa-fusion] " << label << ":";
    for (Ort::ConstNode node : fusion_nodes) {
      std::cerr << " " << node.GetId() << "/" << node.GetOperatorType()
                << "/" << node.GetName();
    }
    std::cerr << std::endl;
  }
}

bool IsFloatTensorValueInfo(Ort::ConstValueInfo value_info) {
  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }

  auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  return type_shape.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

bool IsIntInitializer(Ort::ConstValueInfo value_info) {
  if (!value_info || !value_info.IsConstantInitializer()) {
    return false;
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    return false;
  }

  auto type_shape = value.GetTensorTypeAndShapeInfo();
  auto elem_type = type_shape.GetElementType();
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

std::optional<std::vector<int64_t>> ReadIntInitializer(
    Ort::ConstValueInfo value_info) {
  if (!value_info || !value_info.IsConstantInitializer()) {
    return std::nullopt;
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    return std::nullopt;
  }

  auto type_shape = value.GetTensorTypeAndShapeInfo();
  auto elem_type = type_shape.GetElementType();
  const size_t count = type_shape.GetElementCount();
  std::vector<int64_t> result;
  result.reserve(count);
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* data = value.GetTensorData<int64_t>();
    result.assign(data, data + count);
    return result;
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t* data = value.GetTensorData<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      result.push_back(static_cast<int64_t>(data[i]));
    }
    return result;
  }
  return std::nullopt;
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

  auto shape_from_initializer = ReadIntInitializer(inputs[0]);
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

bool IsSmallMetadataValue(Ort::ConstValueInfo value_info) {
  if (!value_info) {
    return false;
  }

  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }

  auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  auto elem_type = type_shape.GetElementType();
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    return false;
  }

  int64_t element_count = 1;
  for (int64_t dim : type_shape.GetShape()) {
    if (dim < 0) {
      return false;
    }
    element_count *= dim;
  }
  return element_count <= kMaxCpuMetadataElements;
}

bool HasOnlySmallMetadataOutputs(Ort::ConstNode node) {
  std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.empty()) {
    return false;
  }
  for (Ort::ConstValueInfo output : outputs) {
    if (!IsSmallMetadataValue(output)) {
      return false;
    }
  }
  return true;
}

bool IsMetadataOp(Ort::ConstNode node) {
  static const std::unordered_set<std::string> kMetadataOps = {
      "Add",      "And",     "Cast",    "Concat",  "Div",       "Equal",
      "Gather",   "Max",     "Min",     "Mul",     "Or",        "ReduceProd",
      "ReduceSum", "Reshape", "Shape",   "Slice",   "Split",     "Squeeze",
      "Sub",      "Unsqueeze", "Where"};
  return IsOnnxDomain(node.GetDomain()) &&
         kMetadataOps.count(node.GetOperatorType()) != 0;
}

bool IsSmallMetadataInitializer(Ort::ConstValueInfo value_info) {
  return value_info && value_info.IsConstantInitializer() &&
         IsSmallMetadataValue(value_info);
}

std::unordered_set<const OrtNode*> GetCpuPreferredMetadataNodes(
    const OrtGraph& ort_graph, OrtEpGraphSupportInfo& graph_support_info,
    const OrtEpApi& ep_api, const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<size_t>& fused_node_ids) {
  Ort::ConstGraph graph{&ort_graph};
  std::unordered_map<size_t, Ort::ConstNode> node_id_to_node;
  std::unordered_map<size_t, size_t> node_id_to_order;
  for (size_t i = 0; i < all_nodes.size(); ++i) {
    node_id_to_node.emplace(all_nodes[i].GetId(), all_nodes[i]);
    node_id_to_order.emplace(all_nodes[i].GetId(), i);
  }

  std::unordered_set<size_t> provider_nodes;
  std::unordered_map<size_t, Ort::ConstKernelDef> node_to_kernel;
  std::unordered_set<const OrtValueInfo*> cpu_metadata_values;

  for (Ort::ConstNode node : all_nodes) {
    if (fused_node_ids.count(node.GetId()) != 0) {
      continue;
    }

    const OrtKernelDef* kernel_def_ptr = nullptr;
    Ort::ThrowOnError(ep_api.EpGraphSupportInfo_LookUpKernel(
        &graph_support_info, node, &kernel_def_ptr));
    if (kernel_def_ptr == nullptr) {
      continue;
    }

    provider_nodes.insert(node.GetId());
    Ort::ConstKernelDef kernel_def(kernel_def_ptr);
    node_to_kernel.emplace(node.GetId(), kernel_def);

    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
      Ort::ConstValueInfo output = outputs[i];
      if (!output || !IsSmallMetadataValue(output)) {
        continue;
      }
      OrtMemType mem_type = kernel_def.GetOutputMemType(i);
      if (mem_type == OrtMemTypeCPUOutput || mem_type == OrtMemTypeCPUInput) {
        cpu_metadata_values.insert(output);
      }
    }
  }

  auto order_greater = [&](size_t lhs, size_t rhs) {
    return node_id_to_order[lhs] > node_id_to_order[rhs];
  };
  std::priority_queue<size_t, std::vector<size_t>, decltype(order_greater)>
      candidates(order_greater);
  for (const OrtValueInfo* value_info : cpu_metadata_values) {
    Ort::ConstValueInfo output(value_info);
    for (const auto& consumer : output.GetConsumers()) {
      candidates.push(consumer.node.GetId());
    }
  }

  std::unordered_set<size_t> visited;
  std::unordered_set<const OrtNode*> cpu_preferred_nodes;
  while (!candidates.empty()) {
    size_t node_id = candidates.top();
    candidates.pop();
    if (!visited.insert(node_id).second) {
      continue;
    }

    auto node_iter = node_id_to_node.find(node_id);
    if (node_iter == node_id_to_node.end()) {
      continue;
    }
    Ort::ConstNode node = node_iter->second;

    if (provider_nodes.count(node_id) == 0) {
      if (HasOnlySmallMetadataOutputs(node)) {
        std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
        for (Ort::ConstValueInfo output : outputs) {
          cpu_metadata_values.insert(output);
        }
        for (Ort::ConstNode downstream_node : GetOutputNodes(outputs)) {
          candidates.push(downstream_node.GetId());
        }
      }
      continue;
    }

    if (!IsMetadataOp(node) || !HasOnlySmallMetadataOutputs(node)) {
      continue;
    }

    bool place_on_cpu = true;
    if (!IsOnnxOp(node, "Shape")) {
      std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
      Ort::ConstKernelDef kernel_def = node_to_kernel.at(node_id);
      for (size_t i = 0; i < inputs.size(); ++i) {
        Ort::ConstValueInfo input = inputs[i];
        if (!input || IsSmallMetadataInitializer(input)) {
          continue;
        }
        if (cpu_metadata_values.count(input) == 0) {
          place_on_cpu = false;
          break;
        }

        OrtMemType mem_type = kernel_def.GetInputMemType(i);
        if (mem_type == OrtMemTypeCPUInput || mem_type == OrtMemTypeCPUOutput) {
          place_on_cpu = false;
          break;
        }
      }
    }
    if (!place_on_cpu) {
      continue;
    }

    cpu_preferred_nodes.insert(node);
    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
    for (Ort::ConstValueInfo output : outputs) {
      cpu_metadata_values.insert(output);
    }
    for (Ort::ConstNode downstream_node : GetOutputNodes(outputs)) {
      candidates.push(downstream_node.GetId());
    }
  }

  return cpu_preferred_nodes;
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

bool CanFuseConcatSplit(Ort::ConstNode concat_node, Ort::ConstNode split_node,
                        const std::unordered_set<std::string>&
                            graph_output_names,
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
  auto split_sizes = ReadIntInitializer(split_inputs[1]);
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
  fusion_nodes.erase(
      std::unique(fusion_nodes.begin(), fusion_nodes.end(),
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

bool CanFuseSliceConcat(Ort::ConstNode concat_node,
                        const std::unordered_set<std::string>&
                            graph_output_names,
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
    Ort::ValueInfoConsumerProducerInfo producer = concat_input.GetProducerNode();
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
    const bool is_zero_constant_input =
        IsOnnxOp(slice_node, "ConstantOfShape");
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
        if (shape_producer.node &&
            IsOnnxOp(shape_producer.node, "Shape")) {
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

    auto starts = ReadIntInitializer(slice_inputs[1]);
    auto ends = ReadIntInitializer(slice_inputs[2]);
    if (!starts.has_value() || !ends.has_value() ||
        starts->size() != ends->size()) {
      return false;
    }
    std::vector<int64_t> axes(starts->size());
    std::iota(axes.begin(), axes.end(), 0);
    if (slice_inputs.size() > 3 && slice_inputs[3]) {
      auto axes_init = ReadIntInitializer(slice_inputs[3]);
      if (!axes_init.has_value()) {
        return false;
      }
      axes = *axes_init;
    }
    std::vector<int64_t> steps(starts->size(), 1);
    if (slice_inputs.size() > 4 && slice_inputs[4]) {
      auto steps_init = ReadIntInitializer(slice_inputs[4]);
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
  if (bias_shape.has_value() && b_shape.has_value() &&
      !IsBiasShapeForMatMulN(*bias_shape, (*b_shape)[1])) {
    return false;
  }

  return true;
}

bool CanFuseMatMulActivation(Ort::ConstNode matmul_node,
                             Ort::ConstNode activation_node) {
  std::vector<Ort::ConstValueInfo> matmul_inputs = matmul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> matmul_outputs = matmul_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> activation_inputs =
      activation_node.GetInputs();
  std::vector<Ort::ConstValueInfo> activation_outputs =
      activation_node.GetOutputs();
  if (matmul_inputs.size() != 2 || matmul_outputs.size() != 1 ||
      activation_inputs.size() != 1 || activation_outputs.size() != 1 ||
      Name(matmul_outputs[0]) != Name(activation_inputs[0])) {
    return false;
  }

  if (!IsFloatTensorValueInfo(matmul_inputs[0]) ||
      !IsFloatTensorValueInfo(matmul_inputs[1]) ||
      !IsFloatTensorValueInfo(activation_outputs[0])) {
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

  if (a_shape.has_value() && a_shape->back() != (*b_shape)[0]) {
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

bool AddFusionNode(Ort::ConstNode node,
                   const std::unordered_set<size_t>& fused_node_ids,
                   std::unordered_set<size_t>& selected_node_ids,
                   std::vector<Ort::ConstNode>& fusion_nodes);

bool FusionHasNoExternalPathBetweenSelectedNodes(
    const std::vector<Ort::ConstNode>& fusion_nodes,
    const std::unordered_set<size_t>& selected_node_ids);

bool ShapeProducerCanBeInternalized(
    Ort::ConstNode node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& selected_node_ids) {
  if (!IsOnnxOp(node, "Shape") && !IsOnnxOp(node, "Cast") &&
      !IsOnnxOp(node, "Concat") && !IsOnnxOp(node, "Split")) {
    return false;
  }

  for (Ort::ConstValueInfo output : node.GetOutputs()) {
    if (graph_output_names.count(Name(output)) != 0) {
      return false;
    }
    for (const auto& consumer : output.GetConsumers()) {
      if (selected_node_ids.count(consumer.node.GetId()) == 0) {
        return false;
      }
    }
  }
  return true;
}

bool AddPrivateShapeProducerChain(
    Ort::ConstValueInfo value_info,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::unordered_set<size_t>& selected_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  Ort::ConstNode producer = value_info.GetProducerNode().node;
  if (!producer ||
      !ShapeProducerCanBeInternalized(producer, graph_output_names,
                                      selected_node_ids)) {
    return true;
  }
  if (!AddFusionNode(producer, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }
  if (IsOnnxOp(producer, "Shape")) {
    return true;
  }

  for (Ort::ConstValueInfo input : producer.GetInputs()) {
    if (!AddPrivateShapeProducerChain(input, graph_output_names,
                                      fused_node_ids, selected_node_ids,
                                      fusion_nodes)) {
      return false;
    }
  }
  return true;
}

bool CanFuseSharedReshapeMatMulReshape(
    Ort::ConstNode input_reshape,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(input_reshape, "Reshape") ||
      fused_node_ids.count(input_reshape.GetId()) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> reshape_inputs = input_reshape.GetInputs();
  std::vector<Ort::ConstValueInfo> reshape_outputs = input_reshape.GetOutputs();
  if (reshape_inputs.size() != 2 || reshape_outputs.size() != 1 ||
      graph_output_names.count(Name(reshape_outputs[0])) != 0 ||
      !IsFloatTensorValueInfo(reshape_inputs[0]) ||
      !IsFloatTensorValueInfo(reshape_outputs[0])) {
    return false;
  }

  auto input_shape = GetTensorShape(reshape_inputs[0]);
  if (!input_shape.has_value() || input_shape->size() < 3) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> matmul_consumers =
      reshape_outputs[0].GetConsumers();
  if (matmul_consumers.size() != 1 && matmul_consumers.size() != 3) {
    return false;
  }

  std::optional<std::string> output_shape_input_name;
  Ort::ConstValueInfo output_shape_input{nullptr};
  std::unordered_set<size_t> selected_node_ids;
  if (!AddFusionNode(input_reshape, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }

  for (const auto& consumer : matmul_consumers) {
    Ort::ConstNode matmul_node = consumer.node;
    if (consumer.index != 0 || !IsOnnxOp(matmul_node, "MatMul") ||
        fused_node_ids.count(matmul_node.GetId()) != 0) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> matmul_inputs = matmul_node.GetInputs();
    std::vector<Ort::ConstValueInfo> matmul_outputs = matmul_node.GetOutputs();
    if (matmul_inputs.size() != 2 || matmul_outputs.size() != 1 ||
        Name(matmul_inputs[0]) != Name(reshape_outputs[0]) ||
        graph_output_names.count(Name(matmul_outputs[0])) != 0 ||
        !IsFloatTensorValueInfo(matmul_inputs[1]) ||
        !IsFloatTensorValueInfo(matmul_outputs[0])) {
      return false;
    }

    auto weight_shape = GetTensorShape(matmul_inputs[1]);
    if (!weight_shape.has_value() || weight_shape->size() != 2) {
      return false;
    }
    if (input_shape->back() > 0 && (*weight_shape)[0] > 0 &&
        input_shape->back() != (*weight_shape)[0]) {
      return false;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> output_consumers =
        matmul_outputs[0].GetConsumers();
    if (output_consumers.size() != 1 || output_consumers[0].index != 0) {
      return false;
    }

    Ort::ConstNode output_reshape = output_consumers[0].node;
    if (!IsOnnxOp(output_reshape, "Reshape") ||
        fused_node_ids.count(output_reshape.GetId()) != 0) {
      return false;
    }
    std::vector<Ort::ConstValueInfo> output_reshape_inputs =
        output_reshape.GetInputs();
    std::vector<Ort::ConstValueInfo> output_reshape_outputs =
        output_reshape.GetOutputs();
    if (output_reshape_inputs.size() != 2 ||
        output_reshape_outputs.size() != 1 ||
        Name(output_reshape_inputs[0]) != Name(matmul_outputs[0]) ||
        !IsFloatTensorValueInfo(output_reshape_outputs[0])) {
      return false;
    }

    const std::string shape_input_name = Name(output_reshape_inputs[1]);
    if (!output_shape_input_name.has_value()) {
      output_shape_input_name = shape_input_name;
      output_shape_input = output_reshape_inputs[1];
    } else if (*output_shape_input_name != shape_input_name) {
      return false;
    }

    auto output_shape = GetTensorShape(output_reshape_outputs[0]);
    if (output_shape.has_value()) {
      if (output_shape->size() != input_shape->size()) {
        return false;
      }
      for (size_t dim = 0; dim + 1 < output_shape->size(); ++dim) {
        if ((*input_shape)[dim] > 0 && (*output_shape)[dim] > 0 &&
            (*input_shape)[dim] != (*output_shape)[dim]) {
          return false;
        }
      }
      if ((*weight_shape)[1] > 0 && output_shape->back() > 0 &&
          output_shape->back() != (*weight_shape)[1]) {
        return false;
      }
    }

    if (!AddFusionNode(matmul_node, fused_node_ids, selected_node_ids,
                       fusion_nodes) ||
        !AddFusionNode(output_reshape, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }

  if (output_shape_input &&
      !AddPrivateShapeProducerChain(output_shape_input, graph_output_names,
                                    fused_node_ids, selected_node_ids,
                                    fusion_nodes)) {
    return false;
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

std::vector<std::vector<Ort::ConstNode>>
FindSharedReshapeMatMulReshapeFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseSharedReshapeMatMulReshape(node, graph_output_names,
                                           fused_node_ids, fusion_nodes)) {
      continue;
    }
    for (Ort::ConstNode fusion_node : fusion_nodes) {
      fused_node_ids.insert(fusion_node.GetId());
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

std::vector<std::vector<Ort::ConstNode>> FindShapeCastGatherFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode shape_node : all_nodes) {
    if (!IsOnnxOp(shape_node, "Shape") ||
        fused_node_ids.count(shape_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> shape_outputs = shape_node.GetOutputs();
    if (shape_outputs.size() != 1 ||
        graph_output_names.count(shape_outputs[0].GetName()) != 0) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> shape_consumers =
        shape_outputs[0].GetConsumers();
    if (shape_consumers.size() != 1 || shape_consumers[0].index != 0) {
      continue;
    }

    Ort::ConstNode cast_node = shape_consumers[0].node;
    if (!IsOnnxOp(cast_node, "Cast") ||
        fused_node_ids.count(cast_node.GetId()) != 0) {
      continue;
    }
    auto cast_to = GetIntAttribute(cast_node, "to");
    if (!cast_to.has_value() ||
        (*cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
         *cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> cast_outputs = cast_node.GetOutputs();
    if (cast_outputs.size() != 1 ||
        graph_output_names.count(cast_outputs[0].GetName()) != 0) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> cast_consumers =
        cast_outputs[0].GetConsumers();
    if (cast_consumers.size() != 1 || cast_consumers[0].index != 0) {
      continue;
    }

    Ort::ConstNode gather_node = cast_consumers[0].node;
    if (!IsOnnxOp(gather_node, "Gather") ||
        fused_node_ids.count(gather_node.GetId()) != 0 ||
        GetIntAttribute(gather_node, "axis").value_or(0) != 0) {
      continue;
    }

    fusions.push_back({shape_node, cast_node, gather_node});
    fused_node_ids.insert(shape_node.GetId());
    fused_node_ids.insert(cast_node.GetId());
    fused_node_ids.insert(gather_node.GetId());
  }

  return fusions;
}

bool SplitOutputsAreScalarShapeDims(Ort::ConstNode split_node,
                                    size_t shape_rank) {
  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  if (split_inputs.size() > 2 || split_outputs.empty()) {
    return false;
  }

  std::vector<int64_t> split_sizes;
  if (split_inputs.size() == 2) {
    auto sizes = ReadIntInitializer(split_inputs[1]);
    if (!sizes.has_value()) {
      return false;
    }
    split_sizes = *sizes;
  } else {
    if (shape_rank % split_outputs.size() != 0) {
      return false;
    }
    split_sizes.assign(split_outputs.size(),
                       static_cast<int64_t>(shape_rank / split_outputs.size()));
  }

  if (split_sizes.size() != split_outputs.size()) {
    return false;
  }
  return std::all_of(split_sizes.begin(), split_sizes.end(),
                     [](int64_t size) { return size == 1; });
}

bool ShapeSliceSelectsStaticShapeRange(
    Ort::ConstNode slice_node, std::optional<size_t> shape_rank) {
  std::vector<Ort::ConstValueInfo> inputs = slice_node.GetInputs();
  if (inputs.size() < 3 || inputs.size() > 5) {
    return false;
  }
  auto starts = ReadIntInitializer(inputs[1]);
  auto ends = ReadIntInitializer(inputs[2]);
  if (!starts.has_value() || !ends.has_value() || starts->size() != 1 ||
      ends->size() != 1) {
    return false;
  }
  std::vector<int64_t> axes = {0};
  if (inputs.size() > 3 && inputs[3]) {
    auto axes_init = ReadIntInitializer(inputs[3]);
    if (!axes_init.has_value()) {
      return false;
    }
    axes = *axes_init;
  }
  std::vector<int64_t> steps = {1};
  if (inputs.size() > 4 && inputs[4]) {
    auto steps_init = ReadIntInitializer(inputs[4]);
    if (!steps_init.has_value()) {
      return false;
    }
    steps = *steps_init;
  }
  if (axes.size() != 1 || steps.size() != 1 || axes[0] != 0 ||
      steps[0] != 1) {
    return false;
  }
  int64_t start = (*starts)[0];
  int64_t end = (*ends)[0];
  if (shape_rank.has_value()) {
    const int64_t rank = static_cast<int64_t>(*shape_rank);
    start = start < 0 ? start + rank : start;
    end = end < 0 ? end + rank : end;
    start = std::max<int64_t>(0, std::min(start, rank));
    end = std::max<int64_t>(0, std::min(end, rank));
  } else if (start < 0 || end < 0) {
    return false;
  }
  return end > start;
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

bool AddShapeConsumersForShapeCast(
    Ort::ConstNode final_cast_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::unordered_set<size_t>& selected_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!final_cast_node || !IsOnnxOp(final_cast_node, "Cast") ||
      GetIntAttribute(final_cast_node, "to").value_or(0) !=
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      !AddFusionNode(final_cast_node, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> final_cast_outputs =
      final_cast_node.GetOutputs();
  if (final_cast_outputs.size() != 1 ||
      graph_output_names.count(Name(final_cast_outputs[0])) != 0) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> shape_consumers =
      final_cast_outputs[0].GetConsumers();
  if (shape_consumers.empty()) {
    return false;
  }

  for (const auto& consumer : shape_consumers) {
    Ort::ConstNode shape_consumer_node = consumer.node;
    if (consumer.index != 1 ||
        (!IsOnnxOp(shape_consumer_node, "Reshape") &&
         !IsOnnxOp(shape_consumer_node, "Tile")) ||
        !AddFusionNode(shape_consumer_node, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
    std::vector<Ort::ConstValueInfo> inputs =
        shape_consumer_node.GetInputs();
    std::vector<Ort::ConstValueInfo> outputs =
        shape_consumer_node.GetOutputs();
    if (inputs.size() != 2 || outputs.size() != 1) {
      return false;
    }
  }
  return true;
}

bool AddShapeConcatBranch(
    Ort::ConstValueInfo branch_output,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::unordered_set<size_t>& selected_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (graph_output_names.count(Name(branch_output)) != 0) {
    return false;
  }
  Ort::ConstNode branch_node = branch_output.GetProducerNode().node;
  std::vector<Ort::ValueInfoConsumerProducerInfo> concat_consumers =
      branch_output.GetConsumers();
  if (concat_consumers.empty()) {
    return true;
  }

  for (const auto& consumer : concat_consumers) {
    Ort::ConstNode concat_node = consumer.node;
    if (!IsOnnxOp(concat_node, "Concat") ||
        GetIntAttribute(concat_node, "axis").value_or(0) != 0 ||
        !AddFusionNode(concat_node, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }

    bool has_branch_input = false;
    for (Ort::ConstValueInfo input : concat_node.GetInputs()) {
      if (Name(input) == Name(branch_output)) {
        has_branch_input = true;
        continue;
      }
      Ort::ConstNode input_producer = input.GetProducerNode().node;
      if (input_producer && branch_node &&
          input_producer.GetId() == branch_node.GetId()) {
        continue;
      }
      if (!IsIntInitializer(input)) {
        return false;
      }
    }
    std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
    if (!has_branch_input || concat_outputs.size() != 1 ||
        graph_output_names.count(Name(concat_outputs[0])) != 0) {
      return false;
    }
    std::vector<Ort::ValueInfoConsumerProducerInfo> final_cast_consumers =
        concat_outputs[0].GetConsumers();
    if (final_cast_consumers.size() != 1 ||
        final_cast_consumers[0].index != 0) {
      return false;
    }
    if (!AddShapeConsumersForShapeCast(final_cast_consumers[0].node,
                                       graph_output_names, fused_node_ids,
                                       selected_node_ids, fusion_nodes)) {
      return false;
    }
  }
  return true;
}

bool AddShapeReshapeBranch(
    Ort::ConstNode branch_node, std::optional<size_t> shape_rank,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::unordered_set<size_t>& selected_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!AddFusionNode(branch_node, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }

  if (IsOnnxOp(branch_node, "Split")) {
    if (!shape_rank.has_value() ||
        GetIntAttribute(branch_node, "axis").value_or(0) != 0 ||
        !SplitOutputsAreScalarShapeDims(branch_node, *shape_rank)) {
      return false;
    }
  } else if (IsOnnxOp(branch_node, "Slice")) {
    if (!ShapeSliceSelectsStaticShapeRange(branch_node, shape_rank)) {
      return false;
    }
  } else {
    return false;
  }

  std::vector<Ort::ConstValueInfo> branch_outputs = branch_node.GetOutputs();
  if (branch_outputs.empty()) {
    return false;
  }
  for (Ort::ConstValueInfo output : branch_outputs) {
    if (!AddShapeConcatBranch(output, graph_output_names, fused_node_ids,
                              selected_node_ids, fusion_nodes)) {
      return false;
    }
  }
  return true;
}

bool PathReachesSelectedNode(Ort::ConstNode node,
                             const std::unordered_set<size_t>& selected_node_ids,
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

bool CanFuseShapeReshapeGroup(
    Ort::ConstNode shape_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  auto fail = [](const char*) {
    return false;
  };
  if (!IsOnnxOp(shape_node, "Shape") ||
      fused_node_ids.count(shape_node.GetId()) != 0) {
    return fail("not Shape or already fused");
  }

  std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> shape_outputs = shape_node.GetOutputs();
  auto shape_source_shape =
      shape_inputs.size() == 1 ? GetTensorShape(shape_inputs[0]) : std::nullopt;
  std::optional<size_t> shape_source_rank =
      shape_source_shape.has_value()
          ? std::optional<size_t>(shape_source_shape->size())
          : std::nullopt;
  if (shape_inputs.size() != 1 || shape_outputs.size() != 1 ||
      graph_output_names.count(Name(shape_outputs[0])) != 0) {
    return fail("bad Shape input/output");
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> shape_consumers =
      shape_outputs[0].GetConsumers();
  if (shape_consumers.size() != 1 || shape_consumers[0].index != 0) {
    return fail("Shape output consumer is not single input-0 Cast");
  }
  Ort::ConstNode shape_cast_node = shape_consumers[0].node;
  auto shape_cast_to = GetIntAttribute(shape_cast_node, "to");
  if (!IsOnnxOp(shape_cast_node, "Cast") ||
      fused_node_ids.count(shape_cast_node.GetId()) != 0 ||
      (!shape_cast_to.has_value() ||
       (*shape_cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
        *shape_cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))) {
    return fail("Shape Cast invalid or already fused");
  }

  std::vector<Ort::ConstValueInfo> shape_cast_inputs =
      shape_cast_node.GetInputs();
  std::vector<Ort::ConstValueInfo> shape_cast_outputs =
      shape_cast_node.GetOutputs();
  if (shape_cast_inputs.size() != 1 || shape_cast_outputs.size() != 1 ||
      graph_output_names.count(Name(shape_cast_outputs[0])) != 0) {
    return fail("bad Shape Cast input/output");
  }

  std::unordered_set<size_t> selected_node_ids;
  if (!AddFusionNode(shape_node, fused_node_ids, selected_node_ids,
                     fusion_nodes) ||
      !AddFusionNode(shape_cast_node, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return fail("failed to add Shape/Cast");
  }

  size_t reshape_count_before = fusion_nodes.size();
  for (const auto& consumer : shape_cast_outputs[0].GetConsumers()) {
    if (consumer.index != 0 ||
        !AddShapeReshapeBranch(consumer.node, shape_source_rank,
                               graph_output_names, fused_node_ids,
                               selected_node_ids, fusion_nodes)) {
      return fail("shape Cast consumer branch rejected");
    }
  }

  bool has_reshape = false;
  for (Ort::ConstNode node : fusion_nodes) {
    if (IsOnnxOp(node, "Reshape")) {
      has_reshape = true;
      break;
    }
  }
  if (!has_reshape || fusion_nodes.size() == reshape_count_before) {
    return fail("no Reshape in fusion");
  }
  if (!FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                   selected_node_ids)) {
    return fail("external path between selected nodes");
  }

  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindShapeReshapeFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode shape_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseShapeReshapeGroup(shape_node, graph_output_names,
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

int64_t StaticElementCount(Ort::ConstValueInfo value_info) {
  auto shape = GetTensorShape(value_info);
  if (!shape.has_value()) {
    return -1;
  }
  int64_t count = 1;
  for (int64_t dim : *shape) {
    if (dim < 0 || (dim != 0 &&
                    count > std::numeric_limits<int64_t>::max() / dim)) {
      return -1;
    }
    count *= dim;
  }
  return count;
}

bool IsIntTensorValueInfo(Ort::ConstValueInfo value_info) {
  if (!value_info) {
    return false;
  }
  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  auto elem_type = type_info.GetTensorTypeAndShapeInfo().GetElementType();
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

bool HasKnownNonIntTensorType(Ort::ConstValueInfo value_info) {
  if (!value_info) {
    return true;
  }
  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return true;
  }
  auto elem_type = type_info.GetTensorTypeAndShapeInfo().GetElementType();
  return elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED &&
         elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
         elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

bool IsFixedSizeNonStringTensorValueInfo(Ort::ConstValueInfo value_info) {
  if (!value_info) {
    return false;
  }
  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  auto elem_type = type_info.GetTensorTypeAndShapeInfo().GetElementType();
  return elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED &&
         elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
}

bool IsCastOfShapeMetadata(Ort::ConstNode node) {
  if (!IsOnnxOp(node, "Cast")) {
    return false;
  }
  auto cast_to = GetIntAttribute(node, "to");
  if (!cast_to.has_value() ||
      (*cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
       *cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  if (inputs.size() != 1) {
    return false;
  }
  Ort::ConstNode producer = inputs[0].GetProducerNode().node;
  return IsOnnxOp(producer, "Shape");
}

std::optional<size_t> ShapeCastReshapeDimIndexFromSplitOutput(
    Ort::ConstNode split_node, const std::string& output_name) {
  if (GetIntAttribute(split_node, "axis").value_or(0) != 0) {
    return std::nullopt;
  }
  std::vector<Ort::ConstValueInfo> inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> outputs = split_node.GetOutputs();
  if (inputs.empty() || outputs.empty()) {
    return std::nullopt;
  }
  int64_t input_count = StaticElementCount(inputs[0]);
  if (input_count < 0) {
    input_count = static_cast<int64_t>(outputs.size());
  }

  std::vector<int64_t> split_sizes;
  if (inputs.size() > 1 && inputs[1]) {
    auto sizes = ReadIntInitializer(inputs[1]);
    if (!sizes.has_value()) {
      return std::nullopt;
    }
    split_sizes = *sizes;
  } else {
    if (input_count % static_cast<int64_t>(outputs.size()) != 0) {
      return std::nullopt;
    }
    split_sizes.assign(outputs.size(),
                       input_count / static_cast<int64_t>(outputs.size()));
  }
  if (split_sizes.size() != outputs.size()) {
    return std::nullopt;
  }

  int64_t offset = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (Name(outputs[i]) == output_name) {
      if (split_sizes[i] != 1) {
        return std::nullopt;
      }
      return static_cast<size_t>(offset);
    }
    offset += split_sizes[i];
  }
  return std::nullopt;
}

std::optional<size_t> ShapeCastReshapeDimIndexFromSliceOutput(
    Ort::ConstNode slice_node) {
  std::vector<Ort::ConstValueInfo> inputs = slice_node.GetInputs();
  if (inputs.size() < 3 || inputs.size() > 5) {
    return std::nullopt;
  }
  auto starts = ReadIntInitializer(inputs[1]);
  auto ends = ReadIntInitializer(inputs[2]);
  if (!starts.has_value() || !ends.has_value() || starts->size() != 1 ||
      ends->size() != 1) {
    return std::nullopt;
  }

  std::vector<int64_t> axes = {0};
  if (inputs.size() > 3 && inputs[3]) {
    auto axes_init = ReadIntInitializer(inputs[3]);
    if (!axes_init.has_value()) {
      return std::nullopt;
    }
    axes = *axes_init;
  }
  std::vector<int64_t> steps = {1};
  if (inputs.size() > 4 && inputs[4]) {
    auto steps_init = ReadIntInitializer(inputs[4]);
    if (!steps_init.has_value()) {
      return std::nullopt;
    }
    steps = *steps_init;
  }
  if (axes.size() != 1 || steps.size() != 1 || axes[0] != 0 ||
      steps[0] != 1) {
    return std::nullopt;
  }

  int64_t start = (*starts)[0];
  int64_t end = (*ends)[0];
  int64_t input_count = StaticElementCount(inputs[0]);
  if (input_count >= 0) {
    start = start < 0 ? start + input_count : start;
    end = end < 0 ? end + input_count : end;
  } else if (start < 0 || end < 0) {
    return std::nullopt;
  }
  if (end - start != 1 || start < 0) {
    return std::nullopt;
  }
  return static_cast<size_t>(start);
}

bool IsShapeSplitOrSliceScalarOutput(Ort::ConstValueInfo value_info) {
  if (HasKnownNonIntTensorType(value_info)) {
    return false;
  }
  Ort::ConstNode producer = value_info.GetProducerNode().node;
  if (!IsOnnxOp(producer, "Split") && !IsOnnxOp(producer, "Slice")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> inputs = producer.GetInputs();
  if (inputs.empty()) {
    return false;
  }
  Ort::ConstNode shape_cast = inputs[0].GetProducerNode().node;
  if (!IsCastOfShapeMetadata(shape_cast)) {
    return false;
  }
  const int64_t static_element_count = StaticElementCount(value_info);
  if (static_element_count >= 0 && static_element_count != 1) {
    return false;
  }

  std::optional<size_t> dim_index =
      IsOnnxOp(producer, "Split")
          ? ShapeCastReshapeDimIndexFromSplitOutput(producer, Name(value_info))
          : ShapeCastReshapeDimIndexFromSliceOutput(producer);
  return dim_index.has_value() && *dim_index == 0;
}

bool ShapeCastReshapeTermsSupported(Ort::ConstNode concat_node) {
  if (!IsOnnxOp(concat_node, "Concat") ||
      GetIntAttribute(concat_node, "axis").value_or(0) != 0) {
    return false;
  }
  int64_t term_count = 0;
  int64_t infer_count = 0;
  int64_t dynamic_term_count = 0;
  for (Ort::ConstValueInfo input : concat_node.GetInputs()) {
    if (input.IsConstantInitializer()) {
      auto values = ReadIntInitializer(input);
      if (!values.has_value()) {
        return false;
      }
      for (int64_t value : *values) {
        ++term_count;
        if (value == -1) {
          ++infer_count;
        }
      }
      continue;
    }
    if (!IsShapeSplitOrSliceScalarOutput(input)) {
      return false;
    }
    ++term_count;
    ++dynamic_term_count;
  }
  return term_count > 0 && infer_count <= 1 && dynamic_term_count <= 1;
}

bool ShapeCastConcatMatchesTailConstants(
    Ort::ConstNode concat_node, const std::vector<int64_t>& expected_tail) {
  std::vector<Ort::ConstValueInfo> inputs = concat_node.GetInputs();
  if (inputs.empty() || inputs[0].IsConstantInitializer()) {
    return false;
  }

  std::vector<int64_t> constants;
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (!inputs[i].IsConstantInitializer()) {
      return false;
    }
    auto values = ReadIntInitializer(inputs[i]);
    if (!values.has_value()) {
      return false;
    }
    constants.insert(constants.end(), values->begin(), values->end());
  }
  return constants == expected_tail;
}

bool ValueDependsOnFusedNode(Ort::ConstValueInfo value_info,
                             const std::unordered_set<size_t>& fused_node_ids,
                             std::unordered_set<size_t>& visited_node_ids) {
  if (!value_info) {
    return false;
  }
  Ort::ConstNode producer = value_info.GetProducerNode().node;
  if (!producer || !visited_node_ids.insert(producer.GetId()).second) {
    return false;
  }
  if (fused_node_ids.count(producer.GetId()) != 0) {
    return true;
  }
  for (Ort::ConstValueInfo input : producer.GetInputs()) {
    if (ValueDependsOnFusedNode(input, fused_node_ids, visited_node_ids)) {
      return true;
    }
  }
  return false;
}

bool ValueDependsOnTargetNode(Ort::ConstValueInfo value_info,
                              const std::unordered_set<size_t>& target_node_ids,
                              const std::unordered_set<size_t>& ignored_node_ids,
                              std::unordered_set<size_t>& visited_node_ids) {
  if (!value_info) {
    return false;
  }
  Ort::ConstNode producer = value_info.GetProducerNode().node;
  if (!producer || !visited_node_ids.insert(producer.GetId()).second) {
    return false;
  }
  if (target_node_ids.count(producer.GetId()) != 0 &&
      ignored_node_ids.count(producer.GetId()) == 0) {
    return true;
  }
  for (Ort::ConstValueInfo input : producer.GetInputs()) {
    if (ValueDependsOnTargetNode(input, target_node_ids, ignored_node_ids,
                                 visited_node_ids)) {
      return true;
    }
  }
  return false;
}

bool ShapeCastReshapeDependsOnOtherCandidate(
    const std::vector<Ort::ConstNode>& fusion_nodes,
    const std::unordered_set<size_t>& candidate_reshape_node_ids) {
  std::unordered_set<size_t> local_reshape_node_ids;
  for (Ort::ConstNode node : fusion_nodes) {
    if (IsOnnxOp(node, "Reshape")) {
      local_reshape_node_ids.insert(node.GetId());
    }
  }

  for (Ort::ConstNode node : fusion_nodes) {
    if (!IsOnnxOp(node, "Reshape")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    if (inputs.empty()) {
      return true;
    }
    std::unordered_set<size_t> visited_node_ids;
    if (ValueDependsOnTargetNode(inputs[0], candidate_reshape_node_ids,
                                 local_reshape_node_ids, visited_node_ids)) {
      return true;
    }
  }
  return false;
}

bool IsLargePayloadReshapeInput(Ort::ConstValueInfo value_info,
                                Ort::ConstNode shape_concat_node) {
  auto shape = GetTensorShape(value_info);
  if (!shape.has_value() || shape->size() < 3) {
    return false;
  }

  int64_t known_elements = 1;
  for (int64_t dim : *shape) {
    if (dim <= 0) {
      continue;
    }
    if (known_elements > std::numeric_limits<int64_t>::max() / dim) {
      return true;
    }
    known_elements *= dim;
    if (known_elements >= 4096) {
      return true;
    }
  }

  int64_t requested_rank = 0;
  int64_t requested_known_elements = 1;
  for (Ort::ConstValueInfo input : shape_concat_node.GetInputs()) {
    if (!input.IsConstantInitializer()) {
      ++requested_rank;
      continue;
    }
    auto values = ReadIntInitializer(input);
    if (!values.has_value()) {
      continue;
    }
    for (int64_t value : *values) {
      ++requested_rank;
      if (value <= 0) {
        continue;
      }
      if (requested_known_elements >
          std::numeric_limits<int64_t>::max() / value) {
        return true;
      }
      requested_known_elements *= value;
    }
  }
  if (requested_rank >= 3 && requested_known_elements >= 512) {
    return true;
  }
  return false;
}

bool IsAttentionMergeShapeCastReshape(Ort::ConstNode shape_concat_node) {
  if (GetIntAttribute(shape_concat_node, "axis").value_or(0) != 0) {
    return false;
  }

  int64_t requested_rank = 0;
  int64_t dynamic_terms = 0;
  std::vector<int64_t> constant_terms;
  for (Ort::ConstValueInfo input : shape_concat_node.GetInputs()) {
    if (!input.IsConstantInitializer()) {
      ++requested_rank;
      ++dynamic_terms;
      continue;
    }
    auto values = ReadIntInitializer(input);
    if (!values.has_value()) {
      return false;
    }
    requested_rank += static_cast<int64_t>(values->size());
    constant_terms.insert(constant_terms.end(), values->begin(), values->end());
  }

  return requested_rank == 3 && dynamic_terms == 1 &&
         constant_terms.size() == 2 && constant_terms[0] == -1 &&
         constant_terms[1] == 768;
}

bool CanFuseShapeCastReshape(
    Ort::ConstNode concat_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  auto fail = [&](const char* reason) {
    if (std::getenv("MUSA_EP_DEBUG_SHAPE_CAST_RESHAPE") != nullptr) {
      std::cerr << "[shape_cast_reshape.reject] " << concat_node.GetId()
                << "/" << concat_node.GetName() << ": " << reason
                << std::endl;
    }
    return false;
  };
  if (fused_node_ids.count(concat_node.GetId()) != 0 ||
      !ShapeCastReshapeTermsSupported(concat_node)) {
    return fail("terms unsupported or already fused");
  }

  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_outputs.size() != 1 ||
      graph_output_names.count(Name(concat_outputs[0])) != 0) {
    return fail("bad concat output");
  }
  std::vector<Ort::ValueInfoConsumerProducerInfo> concat_consumers =
      concat_outputs[0].GetConsumers();
  if (concat_consumers.size() != 1 || concat_consumers[0].index != 0) {
    return fail("concat output must feed one Cast input-0");
  }

  Ort::ConstNode cast_node = concat_consumers[0].node;
  if (!IsOnnxOp(cast_node, "Cast") ||
      fused_node_ids.count(cast_node.GetId()) != 0 ||
      GetIntAttribute(cast_node, "to").value_or(0) !=
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return fail("final Cast invalid");
  }
  std::vector<Ort::ConstValueInfo> cast_outputs = cast_node.GetOutputs();
  if (cast_outputs.size() != 1 ||
      graph_output_names.count(Name(cast_outputs[0])) != 0) {
    return fail("bad Cast output");
  }

  fusion_nodes = {concat_node, cast_node};
  std::vector<Ort::ValueInfoConsumerProducerInfo> reshape_consumers =
      cast_outputs[0].GetConsumers();
  if (reshape_consumers.empty()) {
    return fail("Cast has no consumers");
  }
  if (reshape_consumers.size() != 1) {
    return fail("multiple Reshape payload copies are not profitable");
  }
  for (const auto& consumer : reshape_consumers) {
    Ort::ConstNode reshape_node = consumer.node;
    if (consumer.index != 1 || !IsOnnxOp(reshape_node, "Reshape") ||
        fused_node_ids.count(reshape_node.GetId()) != 0) {
      return fail("Cast consumer is not unfused Reshape shape input");
    }
    std::vector<Ort::ConstValueInfo> inputs = reshape_node.GetInputs();
    std::vector<Ort::ConstValueInfo> outputs = reshape_node.GetOutputs();
    if (inputs.size() != 2 || outputs.size() != 1 ||
        !IsFixedSizeNonStringTensorValueInfo(inputs[0])) {
      return fail("Reshape input/output unsupported");
    }
    const bool allow_attention_merge =
        std::getenv("MUSA_EP_ENABLE_MERGE_SHAPE_CAST_RESHAPE") != nullptr &&
        IsAttentionMergeShapeCastReshape(concat_node);
    if (IsLargePayloadReshapeInput(inputs[0], concat_node) &&
        !allow_attention_merge) {
      return fail("large payload Reshape should keep kernel alias");
    }
    fusion_nodes.push_back(reshape_node);
  }
  std::unordered_set<size_t> selected_node_ids;
  for (Ort::ConstNode node : fusion_nodes) {
    selected_node_ids.insert(node.GetId());
  }
  if (!FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                   selected_node_ids)) {
    return fail("external path between selected nodes");
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

bool CanFuseShapeCastTranspose(
    Ort::ConstNode concat_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  auto fail = [&](const char* reason) {
    if (std::getenv("MUSA_EP_DEBUG_SHAPE_CAST_TRANSPOSE") != nullptr) {
      std::cerr << "[shape_cast_transpose.reject] " << concat_node.GetId()
                << "/" << concat_node.GetName() << ": " << reason
                << std::endl;
    }
    return false;
  };
  if (fused_node_ids.count(concat_node.GetId()) != 0 ||
      !ShapeCastReshapeTermsSupported(concat_node) ||
      !ShapeCastConcatMatchesTailConstants(concat_node, {-1, 12, 64})) {
    return fail("terms unsupported or already fused");
  }

  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_outputs.size() != 1 ||
      graph_output_names.count(Name(concat_outputs[0])) != 0) {
    return fail("bad concat output");
  }
  std::vector<Ort::ValueInfoConsumerProducerInfo> concat_consumers =
      concat_outputs[0].GetConsumers();
  if (concat_consumers.size() != 1 || concat_consumers[0].index != 0) {
    return fail("concat output must feed one Cast input-0");
  }

  Ort::ConstNode cast_node = concat_consumers[0].node;
  if (!IsOnnxOp(cast_node, "Cast") ||
      fused_node_ids.count(cast_node.GetId()) != 0 ||
      GetIntAttribute(cast_node, "to").value_or(0) !=
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return fail("final Cast invalid");
  }
  std::vector<Ort::ConstValueInfo> cast_outputs = cast_node.GetOutputs();
  if (cast_outputs.size() != 1 ||
      graph_output_names.count(Name(cast_outputs[0])) != 0) {
    return fail("bad Cast output");
  }

  std::unordered_set<size_t> selected_node_ids;
  if (!AddFusionNode(concat_node, fused_node_ids, selected_node_ids,
                     fusion_nodes) ||
      !AddFusionNode(cast_node, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return fail("failed to add Concat/Cast");
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> reshape_consumers =
      cast_outputs[0].GetConsumers();
  if (reshape_consumers.empty()) {
    return fail("Cast has no consumers");
  }
  for (const auto& consumer : reshape_consumers) {
    Ort::ConstNode reshape_node = consumer.node;
    if (consumer.index != 1 || !IsOnnxOp(reshape_node, "Reshape") ||
        fused_node_ids.count(reshape_node.GetId()) != 0) {
      return fail("Cast consumer is not unfused Reshape shape input");
    }
    std::vector<Ort::ConstValueInfo> reshape_inputs =
        reshape_node.GetInputs();
    std::vector<Ort::ConstValueInfo> reshape_outputs =
        reshape_node.GetOutputs();
    if (reshape_inputs.size() != 2 || reshape_outputs.size() != 1 ||
        graph_output_names.count(Name(reshape_outputs[0])) != 0 ||
        !IsFixedSizeNonStringTensorValueInfo(reshape_inputs[0])) {
      return fail("Reshape input/output unsupported");
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> transpose_consumers =
        reshape_outputs[0].GetConsumers();
    if (transpose_consumers.empty()) {
      return fail("Reshape output has no consumers");
    }
    if (!AddFusionNode(reshape_node, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return fail("failed to add Reshape");
    }
    for (const auto& transpose_consumer : transpose_consumers) {
      if (transpose_consumer.index != 0) {
        return fail("Reshape output consumer is not Transpose input-0");
      }
      Ort::ConstNode transpose_node = transpose_consumer.node;
      if (!IsOnnxOp(transpose_node, "Transpose") ||
          fused_node_ids.count(transpose_node.GetId()) != 0) {
        return fail("Transpose invalid or already fused");
      }
      std::vector<Ort::ConstValueInfo> transpose_inputs =
          transpose_node.GetInputs();
      std::vector<Ort::ConstValueInfo> transpose_outputs =
          transpose_node.GetOutputs();
      if (transpose_inputs.size() != 1 || transpose_outputs.size() != 1 ||
          Name(transpose_inputs[0]) != Name(reshape_outputs[0])) {
        return fail("Transpose input/output unsupported");
      }
      if (!AddFusionNode(transpose_node, fused_node_ids, selected_node_ids,
                         fusion_nodes)) {
        return fail("failed to add Transpose");
      }
    }
  }

  if (!FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                   selected_node_ids)) {
    return fail("external path between selected nodes");
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindShapeCastTransposeFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode concat_node : all_nodes) {
    if (!IsOnnxOp(concat_node, "Concat")) {
      continue;
    }
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseShapeCastTranspose(concat_node, graph_output_names,
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

bool CanFuseShapeCastSplit(
    Ort::ConstNode concat_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (fused_node_ids.count(concat_node.GetId()) != 0 ||
      !ShapeCastReshapeTermsSupported(concat_node) ||
      !ShapeCastConcatMatchesTailConstants(concat_node, {-1, 768})) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_outputs.size() != 1 ||
      graph_output_names.count(Name(concat_outputs[0])) != 0) {
    return false;
  }
  std::vector<Ort::ValueInfoConsumerProducerInfo> concat_consumers =
      concat_outputs[0].GetConsumers();
  if (concat_consumers.size() != 1 || concat_consumers[0].index != 0) {
    return false;
  }

  Ort::ConstNode cast_node = concat_consumers[0].node;
  if (!IsOnnxOp(cast_node, "Cast") ||
      fused_node_ids.count(cast_node.GetId()) != 0 ||
      GetIntAttribute(cast_node, "to").value_or(0) !=
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> cast_outputs = cast_node.GetOutputs();
  if (cast_outputs.size() != 1 ||
      graph_output_names.count(Name(cast_outputs[0])) != 0) {
    return false;
  }
  std::vector<Ort::ValueInfoConsumerProducerInfo> reshape_consumers =
      cast_outputs[0].GetConsumers();
  if (reshape_consumers.size() != 1 || reshape_consumers[0].index != 1) {
    return false;
  }

  Ort::ConstNode reshape_node = reshape_consumers[0].node;
  if (!IsOnnxOp(reshape_node, "Reshape") ||
      fused_node_ids.count(reshape_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> reshape_inputs =
      reshape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reshape_outputs =
      reshape_node.GetOutputs();
  if (reshape_inputs.size() != 2 || reshape_outputs.size() != 1 ||
      graph_output_names.count(Name(reshape_outputs[0])) != 0 ||
      !IsFixedSizeNonStringTensorValueInfo(reshape_inputs[0])) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> split_consumers =
      reshape_outputs[0].GetConsumers();
  if (split_consumers.size() != 1 || split_consumers[0].index != 0) {
    return false;
  }
  Ort::ConstNode split_node = split_consumers[0].node;
  if (!IsOnnxOp(split_node, "Split") ||
      fused_node_ids.count(split_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  if (split_inputs.empty() || split_outputs.empty() ||
      Name(split_inputs[0]) != Name(reshape_outputs[0])) {
    return false;
  }
  if (split_inputs.size() > 1 && split_inputs[1] &&
      !split_inputs[1].IsConstantInitializer()) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  if (!AddFusionNode(concat_node, fused_node_ids, selected_node_ids,
                     fusion_nodes) ||
      !AddFusionNode(cast_node, fused_node_ids, selected_node_ids,
                     fusion_nodes) ||
      !AddFusionNode(reshape_node, fused_node_ids, selected_node_ids,
                     fusion_nodes) ||
      !AddFusionNode(split_node, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
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

std::vector<std::vector<Ort::ConstNode>> FindShapeCastSplitFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode concat_node : all_nodes) {
    if (!IsOnnxOp(concat_node, "Concat")) {
      continue;
    }
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseShapeCastSplit(concat_node, graph_output_names,
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

std::vector<std::vector<Ort::ConstNode>> FindShapeCastReshapeFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> candidates;
  for (Ort::ConstNode concat_node : all_nodes) {
    if (!IsOnnxOp(concat_node, "Concat")) {
      continue;
    }
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseShapeCastReshape(concat_node, graph_output_names,
                                 fused_node_ids, fusion_nodes)) {
      continue;
    }
    candidates.push_back(std::move(fusion_nodes));
  }

  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (auto& fusion_nodes : candidates) {
    bool has_already_fused_node = false;
    for (Ort::ConstNode node : fusion_nodes) {
      if (fused_node_ids.count(node.GetId()) != 0) {
        has_already_fused_node = true;
        break;
      }
    }
    if (has_already_fused_node) {
      continue;
    }
    for (Ort::ConstNode node : fusion_nodes) {
      fused_node_ids.insert(node.GetId());
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

bool IsSupportedShapeCast(Ort::ConstNode cast_node) {
  if (!IsOnnxOp(cast_node, "Cast")) {
    return false;
  }
  auto cast_to = GetIntAttribute(cast_node, "to");
  return cast_to.has_value() &&
         (*cast_to == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
          *cast_to == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
}

bool IsShapeCastBranchNode(Ort::ConstNode node, Ort::ConstNode shape_cast_node) {
  if (!IsOnnxOp(node, "Split") && !IsOnnxOp(node, "Slice")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  if (inputs.empty()) {
    return false;
  }
  Ort::ConstNode producer = inputs[0].GetProducerNode().node;
  return producer && producer.GetId() == shape_cast_node.GetId();
}

bool ShapeCastConcatInputSupported(
    Ort::ConstValueInfo input,
    const std::unordered_set<size_t>& branch_node_ids) {
  if (input.IsConstantInitializer()) {
    return ReadIntInitializer(input).has_value();
  }
  Ort::ConstNode producer = input.GetProducerNode().node;
  if (!producer || branch_node_ids.count(producer.GetId()) == 0) {
    return false;
  }
  std::optional<size_t> dim_index =
      IsOnnxOp(producer, "Split")
          ? ShapeCastReshapeDimIndexFromSplitOutput(producer, Name(input))
          : ShapeCastReshapeDimIndexFromSliceOutput(producer);
  return dim_index.has_value();
}

bool ShapeCastSourceBranchSupported(Ort::ConstNode branch_node,
                                    Ort::ConstNode shape_cast_node) {
  if (!IsShapeCastBranchNode(branch_node, shape_cast_node)) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> outputs = branch_node.GetOutputs();
  if (outputs.empty()) {
    return false;
  }
  for (Ort::ConstValueInfo output : outputs) {
    std::optional<size_t> dim_index =
        IsOnnxOp(branch_node, "Split")
            ? ShapeCastReshapeDimIndexFromSplitOutput(branch_node,
                                                      Name(output))
            : ShapeCastReshapeDimIndexFromSliceOutput(branch_node);
    if (!dim_index.has_value()) {
      return false;
    }
  }
  return true;
}

bool ShapeCastSourceFeedsPayloadShape(Ort::ConstValueInfo branch_output) {
  for (const auto& consumer : branch_output.GetConsumers()) {
    Ort::ConstNode concat_node = consumer.node;
    if (consumer.index != 0 || !IsOnnxOp(concat_node, "Concat")) {
      continue;
    }
    if (!ShapeCastReshapeTermsSupported(concat_node)) {
      continue;
    }
    if (ShapeCastConcatMatchesTailConstants(concat_node, {-1, 12, 64}) ||
        ShapeCastConcatMatchesTailConstants(concat_node, {-1, 768})) {
      return true;
    }
  }
  return false;
}

bool CanFuseShapeCastSourceGroup(
    Ort::ConstNode shape_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  auto fail = [&](const char* reason) {
    if (std::getenv("MUSA_EP_DEBUG_SHAPE_CAST_SOURCE") != nullptr) {
      std::cerr << "[shape_cast_source.reject] " << shape_node.GetId()
                << "/" << shape_node.GetName() << ": " << reason
                << std::endl;
    }
    return false;
  };
  if (!IsOnnxOp(shape_node, "Shape") ||
      fused_node_ids.count(shape_node.GetId()) != 0) {
    return fail("not Shape or already fused");
  }
  std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> shape_outputs = shape_node.GetOutputs();
  if (shape_inputs.size() != 1 || shape_outputs.size() != 1 ||
      graph_output_names.count(Name(shape_outputs[0])) != 0) {
    return fail("bad Shape inputs/outputs");
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> shape_consumers =
      shape_outputs[0].GetConsumers();
  if (shape_consumers.size() != 1 || shape_consumers[0].index != 0) {
    return fail("Shape output must feed one Cast input-0");
  }
  Ort::ConstNode shape_cast_node = shape_consumers[0].node;
  if (!IsSupportedShapeCast(shape_cast_node) ||
      fused_node_ids.count(shape_cast_node.GetId()) != 0) {
    return fail("shape Cast invalid or already fused");
  }
  std::vector<Ort::ConstValueInfo> shape_cast_outputs =
      shape_cast_node.GetOutputs();
  if (shape_cast_outputs.size() != 1 ||
      graph_output_names.count(Name(shape_cast_outputs[0])) != 0) {
    return fail("bad shape Cast output");
  }

  std::unordered_set<size_t> selected_node_ids;
  if (!AddFusionNode(shape_node, fused_node_ids, selected_node_ids,
                     fusion_nodes) ||
      !AddFusionNode(shape_cast_node, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return fail("failed to add Shape/Cast");
  }

  bool feeds_payload_shape = false;
  for (const auto& consumer : shape_cast_outputs[0].GetConsumers()) {
    Ort::ConstNode branch_node = consumer.node;
    if (consumer.index != 0 ||
        fused_node_ids.count(branch_node.GetId()) != 0 ||
        !ShapeCastSourceBranchSupported(branch_node, shape_cast_node)) {
      continue;
    }
    bool branch_feeds_payload_shape = false;
    for (Ort::ConstValueInfo output : branch_node.GetOutputs()) {
      if (graph_output_names.count(Name(output)) != 0) {
        return fail("branch output is graph output");
      }
      branch_feeds_payload_shape =
          branch_feeds_payload_shape || ShapeCastSourceFeedsPayloadShape(output);
    }
    if (!branch_feeds_payload_shape) {
      continue;
    }
    feeds_payload_shape = true;
    if (!AddFusionNode(branch_node, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return fail("failed to add Split/Slice");
    }
  }
  if (!feeds_payload_shape) {
    return fail("no payload shape consumer");
  }
  if (!FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                   selected_node_ids)) {
    return fail("external path between selected nodes");
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindShapeCastSourceFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode shape_node : all_nodes) {
    if (!IsOnnxOp(shape_node, "Shape")) {
      continue;
    }
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseShapeCastSourceGroup(shape_node, graph_output_names,
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

bool CanFuseShapeCastConcatGroup(
    Ort::ConstNode shape_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  auto fail = [&](const char* reason) {
    if (std::getenv("MUSA_EP_DEBUG_SHAPE_CAST_CONCAT") != nullptr) {
      std::cerr << "[shape_cast_concat.reject] " << shape_node.GetId()
                << "/" << shape_node.GetName() << ": " << reason
                << std::endl;
    }
    return false;
  };
  if (!IsOnnxOp(shape_node, "Shape") ||
      fused_node_ids.count(shape_node.GetId()) != 0) {
    return fail("not Shape or already fused");
  }
  std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> shape_outputs = shape_node.GetOutputs();
  if (shape_inputs.size() != 1 || shape_outputs.size() != 1 ||
      graph_output_names.count(Name(shape_outputs[0])) != 0) {
    return fail("bad Shape inputs/outputs");
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> shape_consumers =
      shape_outputs[0].GetConsumers();
  if (shape_consumers.size() != 1 || shape_consumers[0].index != 0) {
    return fail("Shape output must feed one Cast input-0");
  }
  Ort::ConstNode shape_cast_node = shape_consumers[0].node;
  if (!IsSupportedShapeCast(shape_cast_node) ||
      fused_node_ids.count(shape_cast_node.GetId()) != 0) {
    return fail("shape Cast invalid or already fused");
  }
  std::vector<Ort::ConstValueInfo> shape_cast_outputs =
      shape_cast_node.GetOutputs();
  if (shape_cast_outputs.size() != 1 ||
      graph_output_names.count(Name(shape_cast_outputs[0])) != 0) {
    return fail("bad shape Cast output");
  }

  std::unordered_set<size_t> selected_node_ids;
  if (!AddFusionNode(shape_node, fused_node_ids, selected_node_ids,
                     fusion_nodes) ||
      !AddFusionNode(shape_cast_node, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return fail("failed to add Shape/Cast");
  }

  std::vector<Ort::ConstNode> branch_nodes;
  std::unordered_set<size_t> branch_node_ids;
  for (const auto& consumer : shape_cast_outputs[0].GetConsumers()) {
    if (fused_node_ids.count(consumer.node.GetId()) != 0) {
      continue;
    }
    if (consumer.index != 0 ||
        !IsShapeCastBranchNode(consumer.node, shape_cast_node) ||
        graph_output_names.count(Name(consumer.node.GetOutputs()[0])) != 0) {
      return fail("shape Cast consumer is not internal Split/Slice");
    }
    if (branch_node_ids.insert(consumer.node.GetId()).second) {
      branch_nodes.push_back(consumer.node);
      if (!AddFusionNode(consumer.node, fused_node_ids, selected_node_ids,
                         fusion_nodes)) {
        return fail("failed to add Split/Slice");
      }
    }
  }
  if (branch_nodes.empty()) {
    return fail("no Split/Slice branches");
  }

  std::vector<Ort::ConstNode> concat_nodes;
  std::unordered_set<size_t> concat_node_ids;
  for (Ort::ConstNode branch_node : branch_nodes) {
    std::vector<Ort::ConstValueInfo> branch_outputs = branch_node.GetOutputs();
    for (Ort::ConstValueInfo branch_output : branch_outputs) {
      if (graph_output_names.count(Name(branch_output)) != 0) {
        return fail("branch output is graph output");
      }
      for (const auto& consumer : branch_output.GetConsumers()) {
        Ort::ConstNode concat_node = consumer.node;
        if (!IsOnnxOp(concat_node, "Concat") ||
            fused_node_ids.count(concat_node.GetId()) != 0) {
          return fail("branch output consumer is not unfused Concat");
        }
        if (concat_node_ids.insert(concat_node.GetId()).second) {
          concat_nodes.push_back(concat_node);
        }
      }
    }
  }
  if (concat_nodes.empty()) {
    return fail("no Concat consumers");
  }

  for (Ort::ConstNode concat_node : concat_nodes) {
    if (GetIntAttribute(concat_node, "axis").value_or(0) != 0) {
      return fail("Concat axis unsupported");
    }
    std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
    std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
    if (concat_inputs.empty() || concat_outputs.size() != 1 ||
        graph_output_names.count(Name(concat_outputs[0])) != 0) {
      return fail("bad Concat inputs/outputs");
    }
    for (Ort::ConstValueInfo input : concat_inputs) {
      if (!ShapeCastConcatInputSupported(input, branch_node_ids)) {
        return fail("Concat input unsupported");
      }
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> final_cast_consumers =
        concat_outputs[0].GetConsumers();
    if (final_cast_consumers.size() != 1 ||
        final_cast_consumers[0].index != 0) {
      return fail("Concat output must feed one final Cast input-0");
    }
    Ort::ConstNode final_cast_node = final_cast_consumers[0].node;
    if (!IsSupportedShapeCast(final_cast_node) ||
        fused_node_ids.count(final_cast_node.GetId()) != 0) {
      return fail("final Cast invalid or already fused");
    }
    std::vector<Ort::ConstValueInfo> final_cast_outputs =
        final_cast_node.GetOutputs();
    if (final_cast_outputs.size() != 1 ||
        graph_output_names.count(Name(final_cast_outputs[0])) != 0) {
      return fail("bad final Cast output");
    }
    std::vector<Ort::ValueInfoConsumerProducerInfo> output_consumers =
        final_cast_outputs[0].GetConsumers();
    if (output_consumers.empty()) {
      return fail("final Cast output has no consumers");
    }
    for (const auto& consumer : output_consumers) {
      if (consumer.index != 1 || !IsOnnxOp(consumer.node, "Reshape") ||
          fused_node_ids.count(consumer.node.GetId()) != 0) {
        return fail("final Cast output consumer is not unfused Reshape shape input");
      }
    }
    if (!AddFusionNode(concat_node, fused_node_ids, selected_node_ids,
                       fusion_nodes) ||
        !AddFusionNode(final_cast_node, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return fail("failed to add Concat/final Cast");
    }
  }

  if (!FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                   selected_node_ids)) {
    return fail("external path between selected nodes");
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindShapeCastConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode shape_node : all_nodes) {
    if (!IsOnnxOp(shape_node, "Shape")) {
      continue;
    }
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseShapeCastConcatGroup(shape_node, graph_output_names,
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

bool AddExpandsForShapeCast(
    Ort::ConstNode final_cast_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::unordered_set<size_t>& selected_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!final_cast_node || !IsOnnxOp(final_cast_node, "Cast") ||
      GetIntAttribute(final_cast_node, "to").value_or(0) !=
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      !AddFusionNode(final_cast_node, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> final_cast_outputs =
      final_cast_node.GetOutputs();
  if (final_cast_outputs.size() != 1 ||
      graph_output_names.count(Name(final_cast_outputs[0])) != 0) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> expand_consumers =
      final_cast_outputs[0].GetConsumers();
  if (expand_consumers.empty()) {
    return false;
  }
  for (const auto& consumer : expand_consumers) {
    Ort::ConstNode output_node = consumer.node;
    if (IsOnnxOp(output_node, "Expand")) {
      if (consumer.index != 1 ||
          !AddFusionNode(output_node, fused_node_ids, selected_node_ids,
                         fusion_nodes)) {
        return false;
      }
      std::vector<Ort::ConstValueInfo> expand_inputs = output_node.GetInputs();
      std::vector<Ort::ConstValueInfo> expand_outputs = output_node.GetOutputs();
      if (expand_inputs.size() != 2 || expand_outputs.size() != 1) {
        return false;
      }
      continue;
    }

    if (IsOnnxOp(output_node, "ConstantOfShape")) {
      if (consumer.index != 0 ||
          !AddFusionNode(output_node, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
        return false;
      }
      std::vector<Ort::ConstValueInfo> cos_inputs = output_node.GetInputs();
      std::vector<Ort::ConstValueInfo> cos_outputs = output_node.GetOutputs();
      if (cos_inputs.size() != 1 || cos_outputs.size() != 1) {
        return false;
      }
      continue;
    }
    return false;
  }
  return true;
}

bool AddShapeExpandBranch(
    Ort::ConstValueInfo branch_output,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::unordered_set<size_t>& selected_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (graph_output_names.count(Name(branch_output)) != 0) {
    return false;
  }
  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
      branch_output.GetConsumers();
  if (consumers.empty()) {
    return false;
  }

  for (const auto& consumer : consumers) {
    if (consumer.index != 0) {
      return false;
    }
    Ort::ConstNode node = consumer.node;
    if (IsOnnxOp(node, "Slice")) {
      continue;
    }
    if (IsOnnxOp(node, "Cast")) {
      if (!AddExpandsForShapeCast(node, graph_output_names, fused_node_ids,
                                  selected_node_ids, fusion_nodes)) {
        return false;
      }
      continue;
    }

    if (IsOnnxOp(node, "Concat")) {
      Ort::ConstNode concat_node = node;
      if (GetIntAttribute(concat_node, "axis").value_or(0) != 0 ||
          !AddFusionNode(concat_node, fused_node_ids, selected_node_ids,
                         fusion_nodes)) {
        return false;
      }

      bool has_branch_input = false;
      for (Ort::ConstValueInfo input : concat_node.GetInputs()) {
        if (Name(input) == Name(branch_output)) {
          has_branch_input = true;
          continue;
        }
        if (!IsIntInitializer(input)) {
          return false;
        }
      }
      std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
      if (!has_branch_input || concat_outputs.size() != 1 ||
          graph_output_names.count(Name(concat_outputs[0])) != 0) {
        return false;
      }
      std::vector<Ort::ValueInfoConsumerProducerInfo> final_cast_consumers =
          concat_outputs[0].GetConsumers();
      if (final_cast_consumers.size() != 1 ||
          final_cast_consumers[0].index != 0) {
        return false;
      }
      if (!AddExpandsForShapeCast(final_cast_consumers[0].node,
                                  graph_output_names, fused_node_ids,
                                  selected_node_ids, fusion_nodes)) {
        return false;
      }
      continue;
    }

    return false;
  }
  return true;
}

bool CanFuseShapeExpandGroup(
    Ort::ConstNode shape_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(shape_node, "Shape")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> shape_outputs = shape_node.GetOutputs();
  if (fused_node_ids.count(shape_node.GetId()) != 0) {
    return false;
  }

  auto shape_source_shape =
      shape_inputs.size() == 1 ? GetTensorShape(shape_inputs[0]) : std::nullopt;
  if (shape_inputs.size() != 1 || shape_outputs.size() != 1 ||
      graph_output_names.count(Name(shape_outputs[0])) != 0) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> shape_consumers =
      shape_outputs[0].GetConsumers();

  std::unordered_set<size_t> selected_node_ids;
  if (!AddFusionNode(shape_node, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }

  Ort::ConstNode shape_cast_node{nullptr};
  std::vector<Ort::ConstValueInfo> shape_cast_outputs;
  for (const auto& consumer : shape_consumers) {
    if (consumer.index != 0) {
      return false;
    }
    Ort::ConstNode consumer_node = consumer.node;
    if (IsOnnxOp(consumer_node, "ConstantOfShape")) {
      std::vector<Ort::ConstValueInfo> cos_inputs = consumer_node.GetInputs();
      std::vector<Ort::ConstValueInfo> cos_outputs = consumer_node.GetOutputs();
      if (cos_inputs.size() != 1 || cos_outputs.size() != 1 ||
          graph_output_names.count(Name(cos_outputs[0])) != 0 ||
          !AddFusionNode(consumer_node, fused_node_ids, selected_node_ids,
                         fusion_nodes)) {
        return false;
      }
      continue;
    }

    if (!IsOnnxOp(consumer_node, "Cast") || shape_cast_node) {
      return false;
    }
    auto shape_cast_to = GetIntAttribute(consumer_node, "to");
    if (fused_node_ids.count(consumer_node.GetId()) != 0 ||
        (!shape_cast_to.has_value() ||
         (*shape_cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
          *shape_cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))) {
      return false;
    }
    shape_cast_outputs = consumer_node.GetOutputs();
    if (shape_cast_outputs.size() != 1 ||
        graph_output_names.count(Name(shape_cast_outputs[0])) != 0 ||
        !AddFusionNode(consumer_node, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
    shape_cast_node = consumer_node;
  }

  bool has_direct_shape_consumer = false;
  if (shape_cast_node) {
    for (const auto& consumer : shape_cast_outputs[0].GetConsumers()) {
      Ort::ConstNode slice_node = consumer.node;
      if (consumer.index != 0) {
        return false;
      }
      if (!IsOnnxOp(slice_node, "Slice")) {
        has_direct_shape_consumer = true;
        continue;
      }
      if (!ShapeSliceSelectsStaticShapeRange(
              slice_node, shape_source_shape.has_value()
                              ? std::optional<size_t>(shape_source_shape->size())
                              : std::nullopt) ||
          !AddFusionNode(slice_node, fused_node_ids, selected_node_ids,
                         fusion_nodes)) {
        return false;
      }
      std::vector<Ort::ConstValueInfo> slice_outputs = slice_node.GetOutputs();
      if (slice_outputs.size() != 1 ||
          !AddShapeExpandBranch(slice_outputs[0], graph_output_names,
                                fused_node_ids, selected_node_ids,
                                fusion_nodes)) {
        return false;
      }
    }
  }
  if (has_direct_shape_consumer &&
      !AddShapeExpandBranch(shape_cast_outputs[0], graph_output_names,
                            fused_node_ids, selected_node_ids, fusion_nodes)) {
    return false;
  }

  bool has_expand = false;
  for (Ort::ConstNode node : fusion_nodes) {
    if (IsOnnxOp(node, "Expand") || IsOnnxOp(node, "ConstantOfShape")) {
      has_expand = true;
      break;
    }
  }
  if (!has_expand ||
      !FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                   selected_node_ids)) {
    return false;
  }

  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindShapeExpandFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode shape_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseShapeExpandGroup(shape_node, graph_output_names, fused_node_ids,
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

bool IsCastToFloat(Ort::ConstNode node) {
  return node && IsOnnxOp(node, "Cast") &&
         GetIntAttribute(node, "to").value_or(0) ==
             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

bool IsBoolTensorValueInfo(Ort::ConstValueInfo value_info) {
  if (!value_info) {
    return false;
  }
  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  return type_info.GetTensorTypeAndShapeInfo().GetElementType() ==
         ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
}

bool TileIsLastDimBoolBroadcast(Ort::ConstNode tile_node) {
  std::vector<Ort::ConstValueInfo> inputs = tile_node.GetInputs();
  if (inputs.size() != 2) {
    return false;
  }
  auto input_shape = GetTensorShape(inputs[0]);
  auto repeats = ReadIntInitializer(inputs[1]);
  if (!repeats.has_value() || repeats->empty()) {
    return false;
  }

  if (input_shape.has_value()) {
    if (input_shape->empty()) {
      return false;
    }
    if (repeats->size() < input_shape->size()) {
      repeats->insert(repeats->begin(), input_shape->size() - repeats->size(),
                      1);
    }
    if (repeats->size() != input_shape->size()) {
      return false;
    }
  }
  if (repeats->back() <= 1) {
    return false;
  }
  return std::all_of(repeats->begin(), repeats->end() - 1,
                     [](int64_t repeat) { return repeat == 1; });
}

bool MulConsumesValue(Ort::ConstNode mul_node, const std::string& value_name,
                      Ort::ConstValueInfo& other_input) {
  if (!IsOnnxOp(mul_node, "Mul")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> inputs = mul_node.GetInputs();
  if (inputs.size() != 2) {
    return false;
  }
  if (Name(inputs[0]) == value_name) {
    other_input = inputs[1];
    return true;
  }
  if (Name(inputs[1]) == value_name) {
    other_input = inputs[0];
    return true;
  }
  return false;
}

Ort::ConstNode SingleConsumerAdd(Ort::ConstNode mul_node) {
  std::vector<Ort::ConstValueInfo> outputs = mul_node.GetOutputs();
  if (outputs.size() != 1) {
    return Ort::ConstNode{nullptr};
  }
  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
      outputs[0].GetConsumers();
  if (consumers.size() != 1 || !IsOnnxOp(consumers[0].node, "Add")) {
    return Ort::ConstNode{nullptr};
  }
  return consumers[0].node;
}

std::optional<std::vector<int64_t>> BroadcastStaticShapeForFusion(
    const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) {
  const size_t rank = std::max(lhs.size(), rhs.size());
  std::vector<int64_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t lhs_dim =
        i < rank - lhs.size() ? 1 : lhs[i - (rank - lhs.size())];
    const int64_t rhs_dim =
        i < rank - rhs.size() ? 1 : rhs[i - (rank - rhs.size())];
    if (lhs_dim == rhs_dim) {
      out[i] = lhs_dim;
    } else if (lhs_dim == 1) {
      out[i] = rhs_dim;
    } else if (rhs_dim == 1) {
      out[i] = lhs_dim;
    } else if (lhs_dim < 0 || rhs_dim < 0) {
      out[i] = lhs_dim < 0 ? rhs_dim : lhs_dim;
    } else {
      return std::nullopt;
    }
  }
  return out;
}

bool ShapesMatchForFusion(const std::vector<int64_t>& actual,
                          const std::vector<int64_t>& expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] >= 0 && expected[i] >= 0 && actual[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

bool CanFuseDirectMaskSelectOutputShapes(Ort::ConstValueInfo mask,
                                         Ort::ConstValueInfo true_input,
                                         Ort::ConstValueInfo false_input,
                                         Ort::ConstValueInfo output) {
  auto mask_shape = GetTensorShape(mask);
  auto true_shape = GetTensorShape(true_input);
  auto false_shape = GetTensorShape(false_input);
  auto output_shape = GetTensorShape(output);
  if (!mask_shape.has_value() || !true_shape.has_value() ||
      !false_shape.has_value() || !output_shape.has_value()) {
    return false;
  }
  auto data_shape =
      BroadcastStaticShapeForFusion(*true_shape, *false_shape);
  if (!data_shape.has_value()) {
    return false;
  }
  auto fused_shape = BroadcastStaticShapeForFusion(*mask_shape, *data_shape);
  return fused_shape.has_value() &&
         ShapesMatchForFusion(*fused_shape, *output_shape);
}

bool CanFuseMaskSelectFromValue(
    Ort::ConstValueInfo mask_value,
    std::span<const Ort::ConstNode> source_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  auto fail = [&](const char* reason) {
    if (std::getenv("MUSA_EP_DEBUG_TILE_MASK_SELECT") != nullptr) {
      std::cerr << "[tile_mask_select.reject] "
                << (mask_value ? Name(mask_value) : std::string("<null>"))
                << ": " << reason << std::endl;
    }
    return false;
  };
  if (!IsBoolTensorValueInfo(mask_value) ||
      graph_output_names.count(Name(mask_value)) != 0) {
    return fail("mask is not bool tensor or graph output");
  }
  if (source_nodes.empty()) {
    auto mask_shape = GetTensorShape(mask_value);
    if (!mask_shape.has_value() || mask_shape->empty() ||
        mask_shape->size() > kMusaMaxBroadcastRank ||
        mask_shape->back() != 1) {
      return fail("direct mask shape unsupported");
    }
  }
  for (Ort::ConstNode node : source_nodes) {
    if (!node || fused_node_ids.count(node.GetId()) != 0) {
      return fail("source node already fused");
    }
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> mask_consumers =
      mask_value.GetConsumers();
  if (mask_consumers.size() != 2) {
    return fail("mask must feed Cast and Not only");
  }

  Ort::ConstNode mask_cast_node{nullptr};
  Ort::ConstNode not_node{nullptr};
  for (const auto& consumer : mask_consumers) {
    if (consumer.index != 0) {
      return fail("mask consumer input index unsupported");
    }
    if (IsCastToFloat(consumer.node)) {
      mask_cast_node = consumer.node;
    } else if (IsOnnxOp(consumer.node, "Not")) {
      not_node = consumer.node;
    } else {
      return fail("mask consumer is not Cast/Not");
    }
  }
  if (!mask_cast_node || !not_node) {
    return fail("missing mask Cast or Not");
  }
  std::vector<Ort::ConstValueInfo> not_outputs = not_node.GetOutputs();
  if (not_outputs.size() != 1 ||
      graph_output_names.count(Name(not_outputs[0])) != 0) {
    return fail("bad Not output");
  }
  std::vector<Ort::ValueInfoConsumerProducerInfo> not_consumers =
      not_outputs[0].GetConsumers();
  if (not_consumers.size() != 1 || not_consumers[0].index != 0 ||
      !IsCastToFloat(not_consumers[0].node)) {
    return fail("Not output must feed one Cast");
  }
  Ort::ConstNode inverse_cast_node = not_consumers[0].node;

  std::unordered_set<size_t> selected_node_ids;
  for (Ort::ConstNode node : source_nodes) {
    if (!AddFusionNode(node, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return fail("failed to add source node");
    }
  }
  for (Ort::ConstNode node : {mask_cast_node, not_node, inverse_cast_node}) {
    if (!AddFusionNode(node, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return fail("failed to add mask Cast/Not node");
    }
  }

  std::vector<Ort::ConstValueInfo> mask_cast_outputs =
      mask_cast_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> inverse_cast_outputs =
      inverse_cast_node.GetOutputs();
  if (mask_cast_outputs.size() != 1 || inverse_cast_outputs.size() != 1 ||
      graph_output_names.count(Name(mask_cast_outputs[0])) != 0 ||
      graph_output_names.count(Name(inverse_cast_outputs[0])) != 0) {
    return fail("bad Cast outputs");
  }
  const std::string mask_float_value = Name(mask_cast_outputs[0]);
  const std::string inverse_value = Name(inverse_cast_outputs[0]);

  std::unordered_set<size_t> used_inverse_mul_ids;
  for (const auto& mask_consumer : mask_cast_outputs[0].GetConsumers()) {
    Ort::ConstNode mask_mul = mask_consumer.node;
    Ort::ConstValueInfo true_input{nullptr};
    if (mask_consumer.index < 0 ||
        !MulConsumesValue(mask_mul, mask_float_value, true_input) ||
        !AddFusionNode(mask_mul, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return fail("mask Cast consumer is not addable Mul");
    }
    Ort::ConstNode add_node = SingleConsumerAdd(mask_mul);
    if (!add_node ||
        !AddFusionNode(add_node, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return fail("mask Mul must feed one Add");
    }

    std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
    std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
    if (add_inputs.size() != 2 || add_outputs.size() != 1) {
      return fail("Add input/output count unsupported");
    }
    Ort::ConstNode lhs = add_inputs[0].GetProducerNode().node;
    Ort::ConstNode rhs = add_inputs[1].GetProducerNode().node;
    Ort::ConstNode inverse_mul =
        lhs.GetId() == mask_mul.GetId() ? rhs : lhs;
    Ort::ConstValueInfo false_input{nullptr};
    if (!MulConsumesValue(inverse_mul, inverse_value, false_input) ||
        !AddFusionNode(inverse_mul, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return fail("inverse branch is not addable Mul");
    }
    if (source_nodes.empty() &&
        !CanFuseDirectMaskSelectOutputShapes(mask_value, true_input,
                                             false_input, add_outputs[0])) {
      return fail("direct output shape unsupported");
    }
    used_inverse_mul_ids.insert(inverse_mul.GetId());
  }

  if (used_inverse_mul_ids.empty()) {
    return fail("no inverse Mul used");
  }
  for (const auto& inverse_consumer : inverse_cast_outputs[0].GetConsumers()) {
    if (used_inverse_mul_ids.count(inverse_consumer.node.GetId()) == 0) {
      return fail("inverse Cast has extra consumers");
    }
  }

  if (!FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                   selected_node_ids)) {
    return fail("external path between selected nodes");
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

bool CanFuseTileMaskSelect(
    Ort::ConstNode tile_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(tile_node, "Tile") ||
      fused_node_ids.count(tile_node.GetId()) != 0 ||
      !TileIsLastDimBoolBroadcast(tile_node)) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> tile_outputs = tile_node.GetOutputs();
  if (tile_outputs.size() != 1) {
    return false;
  }
  const std::array<Ort::ConstNode, 1> source_nodes = {tile_node};
  return CanFuseMaskSelectFromValue(tile_outputs[0], source_nodes,
                                    graph_output_names, fused_node_ids,
                                    fusion_nodes);
}

bool CanFuseDirectMaskSelect(
    Ort::ConstNode mask_cast_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsCastToFloat(mask_cast_node) ||
      fused_node_ids.count(mask_cast_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> inputs = mask_cast_node.GetInputs();
  if (inputs.size() != 1) {
    return false;
  }
  const std::array<Ort::ConstNode, 0> source_nodes = {};
  return CanFuseMaskSelectFromValue(inputs[0], source_nodes,
                                    graph_output_names, fused_node_ids,
                                    fusion_nodes);
}

std::vector<std::vector<Ort::ConstNode>> FindTileMaskSelectFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode tile_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseTileMaskSelect(tile_node, graph_output_names, fused_node_ids,
                               fusion_nodes)) {
      continue;
    }
    for (Ort::ConstNode node : fusion_nodes) {
      fused_node_ids.insert(node.GetId());
    }
    fusions.push_back(std::move(fusion_nodes));
  }

  if (std::getenv("MUSA_EP_DISABLE_DIRECT_MASK_SELECT") == nullptr) {
    for (Ort::ConstNode node : all_nodes) {
      std::vector<Ort::ConstNode> fusion_nodes;
      if (!CanFuseDirectMaskSelect(node, graph_output_names, fused_node_ids,
                                   fusion_nodes)) {
        continue;
      }
      for (Ort::ConstNode fusion_node : fusion_nodes) {
        fused_node_ids.insert(fusion_node.GetId());
      }
      fusions.push_back(std::move(fusion_nodes));
    }
  }

  return fusions;
}

bool ParseRank2ColumnSliceForSliceSumConcat(Ort::ConstNode slice_node,
                                            int64_t& width) {
  if (!IsOnnxOp(slice_node, "Slice")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> inputs = slice_node.GetInputs();
  if (inputs.size() < 3 || inputs.size() > 5 ||
      !IsFloatTensorValueInfo(inputs[0])) {
    return false;
  }
  auto input_shape = GetTensorShape(inputs[0]);
  if (input_shape.has_value() && input_shape->size() != 2) {
    return false;
  }

  auto starts = ReadIntInitializer(inputs[1]);
  auto ends = ReadIntInitializer(inputs[2]);
  if (!starts.has_value() || !ends.has_value() ||
      starts->size() != ends->size()) {
    return false;
  }
  std::vector<int64_t> axes(starts->size());
  std::iota(axes.begin(), axes.end(), 0);
  if (inputs.size() > 3 && inputs[3]) {
    auto axes_init = ReadIntInitializer(inputs[3]);
    if (!axes_init.has_value()) {
      return false;
    }
    axes = *axes_init;
  }
  std::vector<int64_t> steps(starts->size(), 1);
  if (inputs.size() > 4 && inputs[4]) {
    auto steps_init = ReadIntInitializer(inputs[4]);
    if (!steps_init.has_value()) {
      return false;
    }
    steps = *steps_init;
  }
  if (axes.size() != starts->size() || steps.size() != starts->size()) {
    return false;
  }

  bool saw_col_slice = false;
  int64_t start_col = -1;
  int64_t end_col = -1;
  for (size_t i = 0; i < axes.size(); ++i) {
    const int64_t axis = axes[i] < 0 ? axes[i] + 2 : axes[i];
    if (axis < 0 || axis > 1 || steps[i] != 1) {
      return false;
    }
    if (axis == 0) {
      if ((*starts)[i] != 0 ||
          (*ends)[i] < std::numeric_limits<int64_t>::max() / 4) {
        return false;
      }
      continue;
    }
    saw_col_slice = true;
    start_col = (*starts)[i];
    end_col = (*ends)[i];
  }
  if (!saw_col_slice || start_col < 0 || end_col <= start_col) {
    return false;
  }
  width = end_col - start_col;
  return true;
}

bool ValueOnlyConsumedBy(Ort::ConstValueInfo value_info,
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

bool CanFuseSliceSumConcat(
    Ort::ConstNode concat_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(concat_node, "Concat") ||
      fused_node_ids.count(concat_node.GetId()) != 0 ||
      GetIntAttribute(concat_node, "axis").value_or(0) != 1) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_inputs.size() < 2 || concat_outputs.size() != 1 ||
      !IsFloatTensorValueInfo(concat_outputs[0])) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  size_t sum_segment_count = 0;
  size_t slice_count = 0;
  for (Ort::ConstValueInfo concat_input : concat_inputs) {
    Ort::ConstNode producer = concat_input.GetProducerNode().node;
    if (!producer || !IsOnnxOp(producer, "Sum")) {
      if (!IsFloatTensorValueInfo(concat_input)) {
        return false;
      }
      continue;
    }
    if (fused_node_ids.count(producer.GetId()) != 0 ||
        graph_output_names.count(Name(concat_input)) != 0 ||
        !ValueOnlyConsumedBy(concat_input, concat_node)) {
      return false;
    }

    std::vector<Ort::ConstValueInfo> sum_inputs = producer.GetInputs();
    if (sum_inputs.size() < 2) {
      return false;
    }
    int64_t expected_width = -1;
    for (Ort::ConstValueInfo sum_input : sum_inputs) {
      Ort::ConstNode slice_node = sum_input.GetProducerNode().node;
      if (fused_node_ids.count(slice_node.GetId()) != 0 ||
          graph_output_names.count(Name(sum_input)) != 0 ||
          !ValueOnlyConsumedBy(sum_input, producer)) {
        return false;
      }
      int64_t width = -1;
      if (!ParseRank2ColumnSliceForSliceSumConcat(slice_node, width)) {
        return false;
      }
      if (expected_width < 0) {
        expected_width = width;
      } else if (expected_width != width) {
        return false;
      }
      if (!AddFusionNode(slice_node, fused_node_ids, selected_node_ids,
                         fusion_nodes)) {
        return false;
      }
      ++slice_count;
    }
    if (!AddFusionNode(producer, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
    ++sum_segment_count;
  }

  if (sum_segment_count < 2 || slice_count < 8 ||
      slice_count > kMusaSliceSumConcatMaxSlices) {
    fusion_nodes.clear();
    return false;
  }
  if (!AddFusionNode(concat_node, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }
  if (!FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                   selected_node_ids)) {
    fusion_nodes.clear();
    return false;
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindSliceSumConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  if (std::getenv("MUSA_EP_DISABLE_SLICE_SUM_CONCAT") != nullptr) {
    return fusions;
  }
  for (Ort::ConstNode concat_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseSliceSumConcat(concat_node, graph_output_names, fused_node_ids,
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

bool AxesInitializerIs(Ort::ConstValueInfo value_info, int64_t axis) {
  auto axes = ReadIntInitializer(value_info);
  return axes.has_value() && axes->size() == 1 && (*axes)[0] == axis;
}

bool FindUniqueConsumerByInputName(const std::vector<Ort::ConstNode>& all_nodes,
                                   const std::string& value_name,
                                   const char* op_type,
                                   size_t expected_input_index,
                                   Ort::ConstNode& consumer_node) {
  size_t matching_consumers = 0;
  size_t total_consumers = 0;
  Ort::ConstNode matched_node{nullptr};
  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i]) {
        continue;
      }
      if (Name(inputs[i]) != value_name) {
        continue;
      }
      ++total_consumers;
      if (i == expected_input_index && IsOnnxOp(node, op_type)) {
        ++matching_consumers;
        matched_node = node;
      }
    }
  }
  if (total_consumers != 1 || matching_consumers != 1) {
    return false;
  }
  consumer_node = matched_node;
  return true;
}

bool CanFuseMaskedGatherReduce(
    const std::vector<Ort::ConstNode>& all_nodes,
    Ort::ConstNode nonzero_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& fused_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(nonzero_node, "NonZero") ||
      fused_node_ids.count(nonzero_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> nonzero_inputs = nonzero_node.GetInputs();
  std::vector<Ort::ConstValueInfo> nonzero_outputs = nonzero_node.GetOutputs();
  if (nonzero_inputs.size() != 1 || nonzero_outputs.size() != 1 ||
      graph_output_names.count(Name(nonzero_outputs[0])) != 0) {
    return false;
  }

  Ort::ConstNode transpose_node{nullptr};
  if (!FindUniqueConsumerByInputName(all_nodes, Name(nonzero_outputs[0]),
                                     "Transpose", 0, transpose_node)) {
    return false;
  }
  if (fused_node_ids.count(transpose_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> transpose_outputs =
      transpose_node.GetOutputs();
  if (transpose_node.GetInputs().size() != 1 ||
      transpose_outputs.size() != 1 ||
      graph_output_names.count(Name(transpose_outputs[0])) != 0) {
    return false;
  }

  Ort::ConstNode squeeze_node{nullptr};
  if (!FindUniqueConsumerByInputName(all_nodes, Name(transpose_outputs[0]),
                                     "Squeeze", 0, squeeze_node) ||
      fused_node_ids.count(squeeze_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> squeeze_inputs = squeeze_node.GetInputs();
  std::vector<Ort::ConstValueInfo> squeeze_outputs = squeeze_node.GetOutputs();
  if (squeeze_inputs.size() != 2 || squeeze_outputs.size() != 1 ||
      graph_output_names.count(Name(squeeze_outputs[0])) != 0 ||
      !AxesInitializerIs(squeeze_inputs[1], 1)) {
    return false;
  }

  Ort::ConstNode gather_node{nullptr};
  if (!FindUniqueConsumerByInputName(all_nodes, Name(squeeze_outputs[0]),
                                     "Gather", 1, gather_node) ||
      fused_node_ids.count(gather_node.GetId()) != 0 ||
      GetIntAttribute(gather_node, "axis").value_or(0) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gather_outputs = gather_node.GetOutputs();
  if (gather_inputs.size() != 2 || gather_outputs.size() != 1 ||
      graph_output_names.count(Name(gather_outputs[0])) != 0) {
    return false;
  }

  Ort::ConstNode reduce_node{nullptr};
  if (!FindUniqueConsumerByInputName(all_nodes, Name(gather_outputs[0]),
                                     "ReduceProd", 0, reduce_node) &&
      !FindUniqueConsumerByInputName(all_nodes, Name(gather_outputs[0]),
                                     "ReduceMean", 0, reduce_node)) {
    return false;
  }
  if ((!IsOnnxOp(reduce_node, "ReduceProd") &&
       !IsOnnxOp(reduce_node, "ReduceMean")) ||
      fused_node_ids.count(reduce_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> reduce_inputs = reduce_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reduce_outputs = reduce_node.GetOutputs();
  if (reduce_inputs.size() != 2 || reduce_outputs.size() != 1 ||
      !AxesInitializerIs(reduce_inputs[1], 0) ||
      GetIntAttribute(reduce_node, "keepdims").value_or(1) != 0) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  for (Ort::ConstNode node : {nonzero_node, transpose_node, squeeze_node,
                             gather_node, reduce_node}) {
    if (!AddFusionNode(node, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindMaskedGatherReduceFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode nonzero_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseMaskedGatherReduce(all_nodes, nonzero_node, graph_output_names,
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

bool IsReduceSumOrProd(Ort::ConstNode node) {
  return IsOnnxOp(node, "ReduceSum") || IsOnnxOp(node, "ReduceProd");
}

bool ReduceAxesAreLastDim(Ort::ConstNode reduce_node, size_t rank) {
  std::vector<Ort::ConstValueInfo> inputs = reduce_node.GetInputs();
  if (inputs.size() < 2) {
    return false;
  }
  auto axes = ReadIntInitializer(inputs[1]);
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
    if (!AddFusionNode(node, fused_node_ids, selected_node_ids,
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

bool IsBroadcastableToRank3(Ort::ConstValueInfo value_info,
                            const std::vector<int64_t>& output_shape) {
  auto shape = GetTensorShape(value_info);
  if (!shape.has_value() || output_shape.size() != 3 || shape->size() > 3) {
    return false;
  }

  const int64_t rank_delta =
      static_cast<int64_t>(output_shape.size() - shape->size());
  for (size_t dim = 0; dim < output_shape.size(); ++dim) {
    const int64_t input_dim_index = static_cast<int64_t>(dim) - rank_delta;
    if (input_dim_index < 0) {
      continue;
    }
    const int64_t input_dim = (*shape)[static_cast<size_t>(input_dim_index)];
    const int64_t output_dim = output_shape[dim];
    if (input_dim == 1 || input_dim == output_dim || input_dim < 0 ||
        output_dim < 0) {
      continue;
    }
    return false;
  }
  return true;
}

bool KnownShapeIsBroadcastableToRank3(Ort::ConstValueInfo value_info,
                                      const std::vector<int64_t>& output_shape) {
  auto shape = GetTensorShape(value_info);
  return !shape.has_value() || IsBroadcastableToRank3(value_info, output_shape);
}

bool ValueHasSingleConsumer(Ort::ConstValueInfo value_info,
                            Ort::ConstNode expected_consumer,
                            int expected_input_index) {
  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
      value_info.GetConsumers();
  return consumers.size() == 1 &&
         consumers[0].node.GetId() == expected_consumer.GetId() &&
         consumers[0].index == expected_input_index;
}

bool CanFusePowAffineSplitReduce(
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
      !IsIntInitializer(split_inputs[1]) ||
      graph_output_names.count(Name(split_outputs[0])) != 0 ||
      graph_output_names.count(Name(split_outputs[1])) != 0) {
    return false;
  }

  Ort::ConstNode affine_node = split_inputs[0].GetProducerNode().node;
  if ((!IsOnnxOp(affine_node, "Add") && !IsOnnxOp(affine_node, "Sub")) ||
      fused_node_ids.count(affine_node.GetId()) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> affine_inputs = affine_node.GetInputs();
  std::vector<Ort::ConstValueInfo> affine_outputs = affine_node.GetOutputs();
  if (affine_inputs.size() != 2 || affine_outputs.size() != 1 ||
      !IsFloatTensorValueInfo(affine_outputs[0])) {
    return false;
  }

  int pow_input_index = -1;
  Ort::ConstNode pow_node{nullptr};
  for (size_t i = 0; i < affine_inputs.size(); ++i) {
    Ort::ConstNode producer = affine_inputs[i].GetProducerNode().node;
    if (IsOnnxOp(producer, "Pow")) {
      pow_input_index = static_cast<int>(i);
      pow_node = producer;
      break;
    }
  }
  if (!pow_node || fused_node_ids.count(pow_node.GetId()) != 0 ||
      (IsOnnxOp(affine_node, "Sub") && pow_input_index != 0) ||
      !ValueHasSingleConsumer(affine_inputs[static_cast<size_t>(pow_input_index)],
                              affine_node, pow_input_index)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> pow_inputs = pow_node.GetInputs();
  std::vector<Ort::ConstValueInfo> pow_outputs = pow_node.GetOutputs();
  if (pow_inputs.size() != 2 || pow_outputs.size() != 1 ||
      graph_output_names.count(Name(pow_outputs[0])) != 0 ||
      !IsFloatTensorValueInfo(pow_inputs[0]) ||
      !IsFloatTensorValueInfo(pow_inputs[1]) ||
      !IsFloatTensorValueInfo(
          affine_inputs[static_cast<size_t>(1 - pow_input_index)]) ||
      !IsFloatTensorValueInfo(pow_outputs[0]) ||
      !ValueHasSingleConsumer(pow_outputs[0], affine_node, pow_input_index)) {
    return false;
  }

  auto input_shape = GetTensorShape(split_inputs[0]);
  if (input_shape.has_value()) {
    if (input_shape->size() != 3 || (*input_shape)[1] <= 0 ||
        (*input_shape)[2] <= 0 ||
        !KnownShapeIsBroadcastableToRank3(pow_inputs[1], *input_shape) ||
        !KnownShapeIsBroadcastableToRank3(
            affine_inputs[static_cast<size_t>(1 - pow_input_index)],
            *input_shape)) {
      return false;
    }
  }

  auto split_sizes = ReadIntInitializer(split_inputs[1]);
  if (!split_sizes.has_value() || split_sizes->size() != 2 ||
      (*split_sizes)[0] <= 0 || (*split_sizes)[1] <= 0) {
    return false;
  }
  if (input_shape.has_value() &&
      (*split_sizes)[0] + (*split_sizes)[1] != (*input_shape)[1]) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  for (Ort::ConstNode node : {pow_node, affine_node, split_node}) {
    if (!AddFusionNode(node, fused_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }

  for (Ort::ConstValueInfo split_output : split_outputs) {
    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
        split_output.GetConsumers();
    if (consumers.size() != 1 || consumers[0].index != 0 ||
        !(IsOnnxOp(consumers[0].node, "ReduceProd") ||
          IsOnnxOp(consumers[0].node, "ReduceMean"))) {
      return false;
    }
    Ort::ConstNode reduce_node = consumers[0].node;
    std::vector<Ort::ConstValueInfo> reduce_inputs = reduce_node.GetInputs();
    std::vector<Ort::ConstValueInfo> reduce_outputs = reduce_node.GetOutputs();
    if (fused_node_ids.count(reduce_node.GetId()) != 0 ||
        reduce_inputs.size() < 2 ||
        GetIntAttribute(reduce_node, "keepdims").value_or(1) != 0 ||
        !AxesInitializerIs(reduce_inputs[1], 1) ||
        reduce_outputs.size() != 1 ||
        !IsFloatTensorValueInfo(reduce_outputs[0])) {
      return false;
    }
    if (!AddFusionNode(reduce_node, fused_node_ids, selected_node_ids,
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

std::vector<std::vector<Ort::ConstNode>> FindPowAffineSplitReduceFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode split_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFusePowAffineSplitReduce(split_node, graph_output_names,
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

bool CanFuseSplitReduce(Ort::ConstNode split_node,
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
      !IsFloatTensorValueInfo(split_inputs[0]) || !IsIntInitializer(split_inputs[1]) ||
      graph_output_names.count(Name(split_outputs[0])) != 0 ||
      graph_output_names.count(Name(split_outputs[1])) != 0) {
    return false;
  }
  auto input_shape = GetTensorShape(split_inputs[0]);
  if (!input_shape.has_value() || input_shape->size() != 3 ||
      (*input_shape)[1] <= 0 || (*input_shape)[2] <= 0) {
    return false;
  }

  auto split_sizes = ReadIntInitializer(split_inputs[1]);
  if (!split_sizes.has_value() || split_sizes->size() != 2 ||
      (*split_sizes)[0] <= 0 || (*split_sizes)[1] <= 0 ||
      (*split_sizes)[0] + (*split_sizes)[1] != (*input_shape)[1]) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  if (!AddFusionNode(split_node, fused_node_ids, selected_node_ids,
                     fusion_nodes)) {
    return false;
  }
  for (Ort::ConstValueInfo split_output : split_outputs) {
    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
        split_output.GetConsumers();
    if (consumers.size() != 1 || consumers[0].index != 0 ||
        !(IsOnnxOp(consumers[0].node, "ReduceProd") ||
          IsOnnxOp(consumers[0].node, "ReduceMean"))) {
      return false;
    }
    Ort::ConstNode reduce_node = consumers[0].node;
    if (fused_node_ids.count(reduce_node.GetId()) != 0 ||
        reduce_node.GetInputs().size() < 2 ||
        GetIntAttribute(reduce_node, "keepdims").value_or(1) != 0 ||
        !AxesInitializerIs(reduce_node.GetInputs()[1], 1)) {
      return false;
    }
    std::vector<Ort::ConstValueInfo> reduce_outputs = reduce_node.GetOutputs();
    if (reduce_outputs.size() != 1 ||
        !IsFloatTensorValueInfo(reduce_outputs[0])) {
      return false;
    }
    if (!AddFusionNode(reduce_node, fused_node_ids, selected_node_ids,
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
    if (IsLinearActivationNode(add_node) &&
        fused_node_ids.count(add_node.GetId()) == 0 &&
        CanFuseMatMulActivation(matmul_node, add_node)) {
      fusions.push_back({matmul_node, add_node});
      fused_node_ids.insert(matmul_node.GetId());
      fused_node_ids.insert(add_node.GetId());
      continue;
    }

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

    if (!CanFuseMatMulAdd(matmul_node, add_node,
                          matmul_consumers[0].index)) {
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
    std::vector<std::vector<Ort::ConstNode>> concat_split_fusions =
        FindConcatSplitFusions(all_nodes, graph_output_names, fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> concat_matmul_fusions =
        FindConcatMatMulFusions(all_nodes, graph_output_names, fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> slice_concat_fusions =
        FindSliceConcatFusions(all_nodes, graph_output_names, fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> reshape_matmul_reshape_fusions =
        FindSharedReshapeMatMulReshapeFusions(all_nodes, graph_output_names,
                                              fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> shape_cast_source_fusions =
        FindShapeCastSourceFusions(all_nodes, graph_output_names,
                                   fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> shape_cast_transpose_fusions;
    std::vector<std::vector<Ort::ConstNode>> shape_cast_split_fusions;
    std::vector<std::vector<Ort::ConstNode>> shape_reshape_fusions;
    std::vector<std::vector<Ort::ConstNode>> shape_expand_fusions;
    if (!ep->config_.enable_cpu_preferred_metadata) {
      shape_reshape_fusions = FindShapeReshapeFusions(
          all_nodes, graph_output_names, fused_node_ids);
      shape_expand_fusions = FindShapeExpandFusions(
          all_nodes, graph_output_names, fused_node_ids);
    }
    shape_cast_transpose_fusions = FindShapeCastTransposeFusions(
        all_nodes, graph_output_names, fused_node_ids);
    shape_cast_split_fusions = FindShapeCastSplitFusions(
        all_nodes, graph_output_names, fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> shape_cast_concat_fusions =
        FindShapeCastConcatFusions(all_nodes, graph_output_names,
                                   fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> shape_cast_reshape_fusions =
        FindShapeCastReshapeFusions(all_nodes, graph_output_names,
                                    fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> tile_mask_select_fusions =
        FindTileMaskSelectFusions(all_nodes, graph_output_names,
                                  fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> masked_gather_reduce_fusions =
        FindMaskedGatherReduceFusions(all_nodes, graph_output_names,
                                      fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> centered_reduce_fusions =
        FindCenteredReduceFusions(all_nodes, graph_output_names,
                                  fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> pow_affine_split_reduce_fusions =
        FindPowAffineSplitReduceFusions(all_nodes, graph_output_names,
                                        fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> split_reduce_fusions =
        FindSplitReduceFusions(all_nodes, graph_output_names, fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> slice_sum_concat_fusions =
        FindSliceSumConcatFusions(all_nodes, graph_output_names,
                                  fused_node_ids);
    DebugPrintFusions("concat_split", concat_split_fusions);
    DebugPrintFusions("concat_matmul", concat_matmul_fusions);
    DebugPrintFusions("slice_concat", slice_concat_fusions);
    DebugPrintFusions("reshape_matmul_reshape",
                      reshape_matmul_reshape_fusions);
    DebugPrintFusions("shape_cast_source", shape_cast_source_fusions);
    DebugPrintFusions("shape_reshape", shape_reshape_fusions);
    DebugPrintFusions("shape_expand", shape_expand_fusions);
    DebugPrintFusions("shape_cast_concat", shape_cast_concat_fusions);
    DebugPrintFusions("shape_cast_transpose", shape_cast_transpose_fusions);
    DebugPrintFusions("shape_cast_split", shape_cast_split_fusions);
    DebugPrintFusions("shape_cast_reshape", shape_cast_reshape_fusions);
    DebugPrintFusions("tile_mask_select", tile_mask_select_fusions);
    DebugPrintFusions("masked_gather_reduce", masked_gather_reduce_fusions);
    DebugPrintFusions("centered_reduce", centered_reduce_fusions);
    DebugPrintFusions("pow_affine_split_reduce",
                      pow_affine_split_reduce_fusions);
    DebugPrintFusions("split_reduce", split_reduce_fusions);
    DebugPrintFusions("slice_sum_concat", slice_sum_concat_fusions);
    for (const auto& fusion_nodes : concat_split_fusions) {
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

    for (const auto& fusion_nodes : slice_concat_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    for (const auto& fusion_nodes : reshape_matmul_reshape_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    for (const auto& fusion_nodes : shape_cast_source_fusions) {
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
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    for (const auto& fusion_nodes : shape_expand_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    for (const auto& fusion_nodes : shape_cast_concat_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    for (const auto& fusion_nodes : shape_cast_transpose_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    for (const auto& fusion_nodes : shape_cast_split_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    for (const auto& fusion_nodes : shape_cast_reshape_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    for (const auto& fusion_nodes : tile_mask_select_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    for (const auto& fusion_nodes : masked_gather_reduce_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
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

    for (const auto& fusion_nodes : pow_affine_split_reduce_fusions) {
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

    for (const auto& fusion_nodes : slice_sum_concat_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    std::vector<std::vector<Ort::ConstNode>> shape_cast_gather_fusions;
    if (!ep->config_.enable_cpu_preferred_metadata) {
      shape_cast_gather_fusions = FindShapeCastGatherFusions(
          all_nodes, graph_output_names, fused_node_ids);
    }
    std::vector<std::vector<Ort::ConstNode>> gemm_activation_fusions =
        FindGemmActivationFusions(all_nodes, graph_output_names,
                                  fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> fused_gemm_fusions =
        FindFusedGemmFusions(all_nodes, graph_output_names, fused_node_ids);
    DebugPrintFusions("gemm_activation", gemm_activation_fusions);
    DebugPrintFusions("fused_gemm", fused_gemm_fusions);

    for (const auto& fusion_nodes : shape_cast_gather_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = true;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

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

    std::unordered_set<const OrtNode*> cpu_preferred_nodes;
    if (ep->config_.enable_cpu_preferred_metadata) {
      cpu_preferred_nodes = GetCpuPreferredMetadataNodes(
          *ort_graph, *graph_support_info, ep->ep_api_, all_nodes,
          fused_node_ids);
    }

    // Mark non-fused nodes as supported if we have a registered kernel.
    for (const auto& node : all_nodes) {
      if (fused_node_ids.count(node.GetId()) != 0) {
        continue;
      }
      if (cpu_preferred_nodes.count(node) != 0) {
        continue;
      }

      const OrtKernelDef* kernel_def = nullptr;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_LookUpKernel(
          graph_support_info, node, &kernel_def));

      if (kernel_def != nullptr) {
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
