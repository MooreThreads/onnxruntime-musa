// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/split_concat_reorder_fusion.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/split_concat_reorder_impl.h"

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
    throw std::runtime_error("unable to map SplitConcatReorder input " +
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
      throw std::runtime_error(std::string("SplitConcatReorder expects one ") +
                               op_type + " node");
    }
    found = node;
  }
  if (!found) {
    throw std::runtime_error(std::string("SplitConcatReorder expects a ") +
                             op_type + " node");
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

}  // namespace

SplitConcatReorderFusionCompute::SplitConcatReorderFusionCompute(
    size_t input_index, int64_t sequence, int64_t part_count,
    int64_t part_width)
    : input_index(input_index),
      sequence(sequence),
      part_count(part_count),
      part_width(part_width) {}

OrtStatus* SplitConcatReorderFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    if (input_index >= ctx.GetInputCount()) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "SplitConcatReorder source input index out of range");
    }

    Ort::ConstValue input = ctx.GetInput(input_index);
    if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "SplitConcatReorder requires MUSA input");
    }

    auto input_info = input.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "SplitConcatReorder only supports float input");
    }

    const int64_t row_elements = sequence * part_count * part_width;
    if (row_elements <= 0) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT, "SplitConcatReorder invalid static shape");
    }

    std::vector<int64_t> input_shape = input_info.GetShape();
    const int64_t input_elements = NumElementsLocal(input_shape);
    if (input_elements % row_elements != 0) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "SplitConcatReorder input element count does not match reshape");
    }

    const int64_t batch = input_elements / row_elements;
    Ort::UnownedValue output =
        ctx.GetOutput(0, {batch * part_count, sequence, part_width});
    if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "SplitConcatReorder requires MUSA output");
    }

    return LaunchStatus(LaunchMusaSplitConcatReorderFloat(
        input.GetTensorData<float>(), output.GetTensorMutableData<float>(),
        batch, sequence, part_count, part_width, GetComputeStream(ctx)));
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

bool IsSplitConcatReorderFusionGraph(Ort::ConstGraph graph) {
  int reshape_count = 0;
  int split_count = 0;
  int concat_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Reshape")) {
      ++reshape_count;
    } else if (IsOnnxOp(node, "Split")) {
      ++split_count;
    } else if (IsOnnxOp(node, "Concat")) {
      ++concat_count;
    } else {
      return false;
    }
  }
  return reshape_count == 1 && split_count == 1 && concat_count == 1;
}

std::unique_ptr<FusionNodeCompute> CreateSplitConcatReorderFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode reshape_node = FindSingleNode(graph, "Reshape");
  Ort::ConstNode split_node = FindSingleNode(graph, "Split");
  Ort::ConstNode concat_node = FindSingleNode(graph, "Concat");

  std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reshape_outputs = reshape_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  if (reshape_inputs.empty() || reshape_outputs.size() != 1 ||
      split_inputs.empty() || split_outputs.size() < 2 ||
      concat_inputs.size() != split_outputs.size() ||
      Name(split_inputs[0]) != Name(reshape_outputs[0])) {
    throw std::runtime_error("invalid SplitConcatReorder fused graph");
  }

  for (size_t i = 0; i < split_outputs.size(); ++i) {
    if (Name(concat_inputs[i]) != Name(split_outputs[i])) {
      throw std::runtime_error(
          "SplitConcatReorder requires Concat inputs in Split output order");
    }
  }

  auto reshape_shape = GetTensorShape(reshape_outputs[0]);
  if (!reshape_shape.has_value() || reshape_shape->size() != 3 ||
      (*reshape_shape)[1] <= 0 || (*reshape_shape)[2] <= 0) {
    throw std::runtime_error(
        "SplitConcatReorder requires rank-3 Reshape output");
  }

  const int64_t sequence = (*reshape_shape)[1];
  const int64_t packed_width = (*reshape_shape)[2];
  const int64_t part_count = static_cast<int64_t>(split_outputs.size());
  if (packed_width % part_count != 0) {
    throw std::runtime_error(
        "SplitConcatReorder packed width must divide by Split output count");
  }
  const int64_t part_width = packed_width / part_count;
  if (part_width <= 0) {
    throw std::runtime_error("SplitConcatReorder invalid Split part width");
  }

  for (Ort::ConstValueInfo split_output : split_outputs) {
    auto split_shape = GetTensorShape(split_output);
    if (!split_shape.has_value() || split_shape->size() != 3 ||
        (*split_shape)[1] != sequence || (*split_shape)[2] != part_width) {
      throw std::runtime_error(
          "SplitConcatReorder Split output shape mismatch");
    }
  }

  std::unordered_map<std::string, size_t> fused_input_indices =
      FusedInputIndices(fused_node);
  return std::make_unique<SplitConcatReorderFusionCompute>(
      GetFusedInputIndex(fused_input_indices, Name(reshape_inputs[0])),
      sequence, part_count, part_width);
}
