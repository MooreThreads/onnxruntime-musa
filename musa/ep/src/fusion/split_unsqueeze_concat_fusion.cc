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

#include "fusion/split_unsqueeze_concat_fusion.h"

#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/split_unsqueeze_concat_impl.h"

namespace {

constexpr size_t kNoShapeInput = std::numeric_limits<size_t>::max();

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(const Ort::ConstNode& node, const char* op_type) {
  return node.GetOperatorType() == op_type && IsOnnxDomain(node.GetDomain());
}

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
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

std::unordered_map<std::string, size_t> FusedInputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> fused_input_indices;
  std::vector<Ort::ConstValueInfo> fused_inputs = fused_node.GetInputs();
  for (size_t i = 0; i < fused_inputs.size(); ++i) {
    fused_input_indices.emplace(Name(fused_inputs[i]), i);
  }
  return fused_input_indices;
}

size_t GetFusedInputIndex(
    const std::unordered_map<std::string, size_t>& fused_input_indices,
    const std::string& input_name) {
  auto it = fused_input_indices.find(input_name);
  if (it == fused_input_indices.end()) {
    throw std::runtime_error("unable to map SplitUnsqueezeConcat input " +
                             input_name);
  }
  return it->second;
}

Ort::ConstNode FindSingleNode(Ort::ConstGraph graph, const char* op_type) {
  Ort::ConstNode found{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, op_type)) {
      continue;
    }
    if (found) {
      throw std::runtime_error(
          std::string("SplitUnsqueezeConcat expects one ") + op_type + " node");
    }
    found = node;
  }
  if (!found) {
    throw std::runtime_error(std::string("SplitUnsqueezeConcat expects a ") +
                             op_type + " node");
  }
  return found;
}

std::vector<Ort::ConstNode> FindNodes(Ort::ConstGraph graph,
                                      const char* op_type) {
  std::vector<Ort::ConstNode> nodes;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, op_type)) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

Ort::ConstNode FindOptionalSingleNode(Ort::ConstGraph graph,
                                      const char* op_type) {
  Ort::ConstNode found{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, op_type)) {
      continue;
    }
    if (found) {
      throw std::runtime_error(
          std::string("SplitUnsqueezeConcat expects at most one ") + op_type +
          " node");
    }
    found = node;
  }
  return found;
}

int64_t NumElementsLocal(const std::vector<int64_t>& shape) {
  int64_t total = 1;
  for (int64_t dim : shape) {
    total *= dim;
  }
  return total;
}

std::vector<int64_t> ReadRuntimeShapeTensor(Ort::ConstValue value) {
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
    throw std::runtime_error(
        "SplitUnsqueezeConcat requires int32/int64 shape input");
  }
  return result;
}

}  // namespace

SplitUnsqueezeConcatFusionCompute::SplitUnsqueezeConcatFusionCompute(
    size_t input_index, int64_t sequence, int64_t part_count,
    int64_t part_width, bool transpose, size_t shape_input_index)
    : input_index(input_index),
      sequence(sequence),
      part_count(part_count),
      part_width(part_width),
      transpose(transpose),
      shape_input_index(shape_input_index) {}

OrtStatus* SplitUnsqueezeConcatFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    if (input_index >= ctx.GetInputCount()) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "SplitUnsqueezeConcat source input index out of range");
    }

    Ort::ConstValue input = ctx.GetInput(input_index);
    if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "SplitUnsqueezeConcat requires MUSA input");
    }

    auto input_info = input.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "SplitUnsqueezeConcat only supports float input");
    }

    std::vector<int64_t> input_shape = input_info.GetShape();
    const int64_t input_elements = NumElementsLocal(input_shape);

    int64_t runtime_batch = -1;
    int64_t runtime_sequence = sequence;
    int64_t runtime_part_width = part_width;
    int64_t runtime_packed_width = -1;
    if (shape_input_index != kNoShapeInput &&
        (runtime_sequence <= 0 || runtime_part_width <= 0)) {
      if (shape_input_index >= ctx.GetInputCount()) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "SplitUnsqueezeConcat shape input index out of range");
      }
      Ort::ConstValue shape_input = ctx.GetInput(shape_input_index);
      if (IsGpuMemory(shape_input.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "SplitUnsqueezeConcat requires CPU shape input");
      }
      std::vector<int64_t> target_shape = ReadRuntimeShapeTensor(shape_input);
      if (target_shape.size() != 3) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "SplitUnsqueezeConcat requires rank-3 reshape target");
      }

      int unknown_dim = -1;
      int64_t known_product = 1;
      for (size_t i = 0; i < target_shape.size(); ++i) {
        int64_t dim = target_shape[i];
        if (dim == 0) {
          if (i >= input_shape.size()) {
            return Ort::GetApi().CreateStatus(
                ORT_INVALID_ARGUMENT,
                "SplitUnsqueezeConcat invalid zero reshape dimension");
          }
          dim = input_shape[i];
          target_shape[i] = dim;
        } else if (dim == -1) {
          if (unknown_dim >= 0) {
            return Ort::GetApi().CreateStatus(
                ORT_INVALID_ARGUMENT,
                "SplitUnsqueezeConcat multiple inferred reshape dimensions");
          }
          unknown_dim = static_cast<int>(i);
          continue;
        }

        if (dim <= 0) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "SplitUnsqueezeConcat invalid reshape target dimension");
        }
        known_product *= dim;
      }
      if (unknown_dim >= 0) {
        if (known_product <= 0 || input_elements % known_product != 0) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "SplitUnsqueezeConcat cannot infer reshape dimension");
        }
        target_shape[static_cast<size_t>(unknown_dim)] =
            input_elements / known_product;
      }

      runtime_batch = target_shape[0];
      runtime_sequence = target_shape[1];
      runtime_packed_width = target_shape[2];
      if (runtime_packed_width % part_count != 0) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "SplitUnsqueezeConcat packed width does not divide by parts");
      }
      runtime_part_width = runtime_packed_width / part_count;
    } else if (input_shape.size() == 3) {
      runtime_batch = input_shape[0];
      runtime_sequence = input_shape[1];
      runtime_packed_width = input_shape[2];
      if (runtime_packed_width % part_count != 0) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "SplitUnsqueezeConcat packed width does not divide by parts");
      }
      runtime_part_width = runtime_packed_width / part_count;
    }

    if (runtime_sequence <= 0 || runtime_part_width <= 0) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "SplitUnsqueezeConcat cannot resolve runtime shape");
    }

    const int64_t row_elements =
        runtime_sequence * part_count * runtime_part_width;
    if (row_elements <= 0) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT, "SplitUnsqueezeConcat invalid static shape");
    }

    if (input_elements % row_elements != 0) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "SplitUnsqueezeConcat input element count does not match reshape");
    }

    const int64_t batch =
        runtime_batch > 0 ? runtime_batch : input_elements / row_elements;
    if (batch * row_elements != input_elements) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "SplitUnsqueezeConcat resolved shape does not match input elements");
    }
    std::vector<int64_t> output_shape =
        transpose ? std::vector<int64_t>{part_count, batch, runtime_part_width,
                                         runtime_sequence}
                  : std::vector<int64_t>{part_count, batch, runtime_sequence,
                                         runtime_part_width};
    Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
    if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "SplitUnsqueezeConcat requires MUSA output");
    }

    return LaunchStatus(LaunchMusaSplitUnsqueezeConcatFloat(
        input.GetTensorData<float>(), output.GetTensorMutableData<float>(),
        batch, runtime_sequence, part_count, runtime_part_width, transpose,
        GetComputeStream(ctx)));
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

bool IsSplitUnsqueezeConcatFusionGraph(Ort::ConstGraph graph) {
  int reshape_count = 0;
  int split_count = 0;
  int unsqueeze_count = 0;
  int concat_count = 0;
  int transpose_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Reshape")) {
      ++reshape_count;
    } else if (IsOnnxOp(node, "Split")) {
      ++split_count;
    } else if (IsOnnxOp(node, "Unsqueeze")) {
      ++unsqueeze_count;
    } else if (IsOnnxOp(node, "Concat")) {
      ++concat_count;
    } else if (IsOnnxOp(node, "Transpose")) {
      ++transpose_count;
    } else {
      return false;
    }
  }
  return reshape_count <= 1 && split_count == 1 && unsqueeze_count >= 2 &&
         concat_count == 1 && transpose_count <= 1;
}

std::unique_ptr<FusionNodeCompute> CreateSplitUnsqueezeConcatFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode reshape_node = FindOptionalSingleNode(graph, "Reshape");
  Ort::ConstNode split_node = FindSingleNode(graph, "Split");
  Ort::ConstNode concat_node = FindSingleNode(graph, "Concat");
  Ort::ConstNode transpose_node = FindOptionalSingleNode(graph, "Transpose");
  std::vector<Ort::ConstNode> unsqueeze_nodes = FindNodes(graph, "Unsqueeze");

  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (split_inputs.empty() || split_inputs.size() > 2 ||
      split_outputs.size() < 2 ||
      unsqueeze_nodes.size() != split_outputs.size() ||
      concat_inputs.size() != split_outputs.size() ||
      concat_outputs.size() != 1) {
    throw std::runtime_error("invalid SplitUnsqueezeConcat fused graph");
  }

  Ort::ConstValueInfo packed_value = split_inputs[0];
  Ort::ConstValueInfo source_value = split_inputs[0];
  std::string shape_input_name;
  std::optional<std::vector<int64_t>> reshape_target_shape;
  if (reshape_node) {
    std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
    std::vector<Ort::ConstValueInfo> reshape_outputs =
        reshape_node.GetOutputs();
    if (reshape_inputs.empty() || reshape_outputs.size() != 1 ||
        Name(split_inputs[0]) != Name(reshape_outputs[0])) {
      throw std::runtime_error("invalid SplitUnsqueezeConcat Reshape input");
    }
    packed_value = reshape_outputs[0];
    source_value = reshape_inputs[0];
    if (reshape_inputs.size() >= 2) {
      shape_input_name = Name(reshape_inputs[1]);
      reshape_target_shape = ReadIntInitializerNoLimit(reshape_inputs[1]);
    }
  }

  int64_t split_axis = 0;
  if (!NormalizeAxis(GetIntAttribute(split_node, "axis").value_or(0), 3,
                     split_axis) ||
      split_axis != 2) {
    throw std::runtime_error("SplitUnsqueezeConcat requires Split axis 2");
  }

  int64_t concat_axis = 0;
  auto concat_axis_attr = GetIntAttribute(concat_node, "axis");
  if (!concat_axis_attr.has_value() ||
      !NormalizeAxis(*concat_axis_attr, 4, concat_axis) || concat_axis != 0) {
    throw std::runtime_error("SplitUnsqueezeConcat requires Concat axis 0");
  }

  const int64_t part_count = static_cast<int64_t>(split_outputs.size());
  int64_t batch = -1;
  int64_t sequence = -1;
  int64_t packed_width = -1;
  int64_t part_width_from_outputs = -1;

  if (reshape_target_shape.has_value()) {
    if (reshape_target_shape->size() != 3) {
      throw std::runtime_error(
          "SplitUnsqueezeConcat requires rank-3 reshape target");
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
      throw std::runtime_error(
          "SplitUnsqueezeConcat Split input shape mismatch");
    }
    if ((*packed_shape)[0] > 0) {
      if (batch > 0 && batch != (*packed_shape)[0]) {
        throw std::runtime_error("SplitUnsqueezeConcat batch mismatch");
      }
      batch = (*packed_shape)[0];
    }
    if ((*packed_shape)[1] > 0) {
      if (sequence > 0 && sequence != (*packed_shape)[1]) {
        throw std::runtime_error("SplitUnsqueezeConcat sequence mismatch");
      }
      sequence = (*packed_shape)[1];
    }
    if ((*packed_shape)[2] > 0) {
      if (packed_width > 0 && packed_width != (*packed_shape)[2]) {
        throw std::runtime_error("SplitUnsqueezeConcat packed width mismatch");
      }
      packed_width = (*packed_shape)[2];
    }
  }

  auto first_split_shape = GetTensorShape(split_outputs[0]);
  if (first_split_shape.has_value()) {
    if (first_split_shape->size() != 3) {
      throw std::runtime_error(
          "SplitUnsqueezeConcat requires rank-3 Split outputs");
    }
    if ((*first_split_shape)[0] > 0) {
      if (batch > 0 && batch != (*first_split_shape)[0]) {
        throw std::runtime_error("SplitUnsqueezeConcat batch mismatch");
      }
      batch = (*first_split_shape)[0];
    }
    if ((*first_split_shape)[1] > 0) {
      if (sequence > 0 && sequence != (*first_split_shape)[1]) {
        throw std::runtime_error("SplitUnsqueezeConcat sequence mismatch");
      }
      sequence = (*first_split_shape)[1];
    }
    if ((*first_split_shape)[2] > 0) {
      part_width_from_outputs = (*first_split_shape)[2];
    }
  }

  std::vector<int64_t> split_sizes;
  int64_t part_width = part_width_from_outputs;
  if (split_inputs.size() == 2) {
    auto split_initializer = ReadIntInitializerNoLimit(split_inputs[1]);
    if (!split_initializer.has_value() ||
        split_initializer->size() != split_outputs.size()) {
      throw std::runtime_error(
          "SplitUnsqueezeConcat requires constant Split sizes");
    }
    split_sizes = std::move(*split_initializer);
    if (split_sizes.empty() || split_sizes[0] <= 0) {
      throw std::runtime_error("SplitUnsqueezeConcat invalid Split part width");
    }
    part_width = split_sizes[0];
  } else if (packed_width > 0) {
    if (packed_width % part_count != 0) {
      throw std::runtime_error(
          "SplitUnsqueezeConcat packed width must divide by Split outputs");
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
    throw std::runtime_error(
        "SplitUnsqueezeConcat Split size/output shape mismatch");
  }
  int64_t split_total = 0;
  for (int64_t split_size : split_sizes) {
    if (split_size <= 0 || (part_width > 0 && split_size != part_width)) {
      throw std::runtime_error(
          "SplitUnsqueezeConcat requires equal Split sizes");
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
    throw std::runtime_error(
        "SplitUnsqueezeConcat Split sizes do not match packed width");
  }

  std::unordered_set<size_t> used_unsqueeze_nodes;
  for (size_t i = 0; i < split_outputs.size(); ++i) {
    Ort::ConstValueInfo split_output = split_outputs[i];
    Ort::ConstNode matched_unsqueeze{nullptr};
    for (Ort::ConstNode unsqueeze_node : unsqueeze_nodes) {
      std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
          unsqueeze_node.GetInputs();
      if (!unsqueeze_inputs.empty() &&
          Name(unsqueeze_inputs[0]) == Name(split_output)) {
        matched_unsqueeze = unsqueeze_node;
        break;
      }
    }
    if (!matched_unsqueeze ||
        !used_unsqueeze_nodes.insert(matched_unsqueeze.GetId()).second) {
      throw std::runtime_error(
          "SplitUnsqueezeConcat requires one Unsqueeze per Split output");
    }

    auto axes = ReadUnsqueezeAxes(matched_unsqueeze);
    int64_t unsqueeze_axis = 0;
    if (!axes.has_value() || axes->size() != 1 ||
        !NormalizeAxis((*axes)[0], 4, unsqueeze_axis) || unsqueeze_axis != 0) {
      throw std::runtime_error(
          "SplitUnsqueezeConcat requires Unsqueeze axis 0");
    }

    std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
        matched_unsqueeze.GetOutputs();
    if (unsqueeze_outputs.size() != 1 ||
        Name(concat_inputs[i]) != Name(unsqueeze_outputs[0])) {
      throw std::runtime_error(
          "SplitUnsqueezeConcat requires Concat inputs in Split output order");
    }

    auto split_shape = GetTensorShape(split_output);
    auto unsqueeze_shape = GetTensorShape(unsqueeze_outputs[0]);
    if (split_shape.has_value()) {
      if (split_shape->size() != 3 ||
          ((*split_shape)[0] > 0 && batch > 0 && (*split_shape)[0] != batch) ||
          ((*split_shape)[1] > 0 && sequence > 0 &&
           (*split_shape)[1] != sequence) ||
          ((*split_shape)[2] > 0 && part_width > 0 &&
           (*split_shape)[2] != part_width)) {
        throw std::runtime_error("SplitUnsqueezeConcat Split shape mismatch");
      }
    }
    if (unsqueeze_shape.has_value()) {
      if (unsqueeze_shape->size() != 4 ||
          ((*unsqueeze_shape)[0] > 0 && (*unsqueeze_shape)[0] != 1) ||
          ((*unsqueeze_shape)[1] > 0 && batch > 0 &&
           (*unsqueeze_shape)[1] != batch) ||
          ((*unsqueeze_shape)[2] > 0 && sequence > 0 &&
           (*unsqueeze_shape)[2] != sequence) ||
          ((*unsqueeze_shape)[3] > 0 && part_width > 0 &&
           (*unsqueeze_shape)[3] != part_width)) {
        throw std::runtime_error(
            "SplitUnsqueezeConcat Unsqueeze shape mismatch");
      }
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
      throw std::runtime_error(
          "SplitUnsqueezeConcat Concat output shape mismatch");
    }
  }

  const bool transpose = static_cast<bool>(transpose_node);
  if (transpose) {
    std::vector<Ort::ConstValueInfo> transpose_inputs =
        transpose_node.GetInputs();
    std::vector<Ort::ConstValueInfo> transpose_outputs =
        transpose_node.GetOutputs();
    auto perm = GetIntsAttribute(transpose_node, "perm");
    if (transpose_inputs.size() != 1 || transpose_outputs.size() != 1 ||
        Name(transpose_inputs[0]) != Name(concat_outputs[0]) ||
        !perm.has_value() || *perm != std::vector<int64_t>({0, 1, 3, 2})) {
      throw std::runtime_error(
          "SplitUnsqueezeConcat requires Transpose perm [0,1,3,2]");
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
        throw std::runtime_error(
            "SplitUnsqueezeConcat Transpose output shape mismatch");
      }
    }
  }

  std::unordered_map<std::string, size_t> fused_input_indices =
      FusedInputIndices(fused_node);
  const size_t shape_input_index =
      shape_input_name.empty()
          ? kNoShapeInput
          : GetFusedInputIndex(fused_input_indices, shape_input_name);
  return std::make_unique<SplitUnsqueezeConcatFusionCompute>(
      GetFusedInputIndex(fused_input_indices, Name(source_value)), sequence,
      part_count, part_width, transpose, shape_input_index);
}
