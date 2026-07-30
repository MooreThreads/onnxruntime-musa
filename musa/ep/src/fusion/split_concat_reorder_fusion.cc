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

#include "fusion/split_concat_reorder_fusion.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
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
    throw std::runtime_error("unable to map SplitConcat input " + input_name);
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
      throw std::runtime_error(std::string("SplitConcat expects one ") +
                               op_type + " node");
    }
    found = node;
  }
  if (!found) {
    throw std::runtime_error(std::string("SplitConcat expects a ") + op_type +
                             " node");
  }
  return found;
}

}  // namespace

SplitConcatFusionCompute::SplitConcatFusionCompute(size_t input_index,
                                                   int64_t part_count,
                                                   bool transpose_output,
                                                   int64_t sequence,
                                                   int64_t part_width)
    : input_index(input_index),
      part_count(part_count),
      transpose_output(transpose_output),
      sequence(sequence),
      part_width(part_width) {}

OrtStatus* SplitConcatFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    if (input_index >= ctx.GetInputCount()) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT, "SplitConcat source input index out of range");
    }

    Ort::ConstValue input = ctx.GetInput(input_index);
    if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "SplitConcat requires MUSA input");
    }

    auto input_info = input.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "SplitConcat only supports float input");
    }

    std::vector<int64_t> input_shape = input_info.GetShape();
    int64_t batch = 0;
    int64_t runtime_sequence = 0;
    int64_t runtime_part_width = 0;
    int64_t trailing_elements = 1;
    if (input_shape.size() >= 3) {
      batch = input_shape[0];
      runtime_sequence = input_shape[1];
      if (input_shape[2] > 0 && part_count > 0 &&
          input_shape[2] % part_count == 0) {
        runtime_part_width = input_shape[2] / part_count;
      }
      for (size_t i = 3; i < input_shape.size(); ++i) {
        if (input_shape[i] < 0) {
          trailing_elements = 0;
          break;
        }
        trailing_elements *= input_shape[i];
      }
    } else if (input_shape.size() == 2 && sequence > 0 && part_width > 0) {
      const int64_t row_width = sequence * part_count * part_width;
      if (row_width > 0 && input_shape[0] >= 0 && input_shape[1] > 0 &&
          input_shape[1] % row_width == 0) {
        batch = input_shape[0] * (input_shape[1] / row_width);
        runtime_sequence = sequence;
        runtime_part_width = part_width;
      }
    }
    if (batch < 0 || runtime_sequence < 0 || runtime_part_width <= 0 ||
        trailing_elements < 0 || (transpose_output && trailing_elements != 1)) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "SplitConcat requires rank-3 input with equal axis-2 parts");
    }

    if ((sequence > 0 && sequence != runtime_sequence) ||
        (part_width > 0 && part_width != runtime_part_width)) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "SplitConcat runtime shape disagrees with static Split sizes");
    }

    std::vector<int64_t> output_shape;
    if (input_shape.size() == 2 || transpose_output) {
      output_shape = {batch * part_count, runtime_part_width, runtime_sequence};
      if (!transpose_output) {
        output_shape = {batch * part_count, runtime_sequence,
                        runtime_part_width};
      }
    } else {
      output_shape = input_shape;
      output_shape[0] = batch * part_count;
      output_shape[2] = runtime_part_width;
    }
    Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
    if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "SplitConcat requires MUSA output");
    }

    return LaunchStatus(LaunchMusaSplitConcatReorderFloat(
        input.GetTensorData<float>(), output.GetTensorMutableData<float>(),
        batch, runtime_sequence, part_count, runtime_part_width,
        trailing_elements, transpose_output, GetComputeStream(ctx)));
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

bool IsSplitConcatFusionGraph(Ort::ConstGraph graph) {
  int reshape_count = 0;
  int split_count = 0;
  int concat_count = 0;
  int transpose_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Reshape")) {
      ++reshape_count;
    } else if (IsOnnxOp(node, "Split")) {
      ++split_count;
    } else if (IsOnnxOp(node, "Concat")) {
      ++concat_count;
    } else if (IsOnnxOp(node, "Transpose")) {
      ++transpose_count;
    } else {
      return false;
    }
  }
  return reshape_count <= 1 && split_count == 1 && concat_count == 1 &&
         transpose_count <= 1;
}

std::unique_ptr<FusionNodeCompute> CreateSplitConcatFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode split_node = FindSingleNode(graph, "Split");
  Ort::ConstNode concat_node = FindSingleNode(graph, "Concat");
  Ort::ConstNode reshape_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Reshape")) {
      reshape_node = node;
    }
  }

  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  if (split_inputs.empty() || split_outputs.size() < 2 ||
      concat_inputs.size() != split_outputs.size()) {
    throw std::runtime_error("invalid SplitConcat fused graph");
  }

  for (size_t i = 0; i < split_outputs.size(); ++i) {
    if (Name(concat_inputs[i]) != Name(split_outputs[i])) {
      throw std::runtime_error(
          "SplitConcat requires Concat inputs in Split output order");
    }
  }

  const int64_t part_count = static_cast<int64_t>(split_outputs.size());
  int64_t sequence = 0;
  int64_t part_width = 0;
  if (split_inputs.size() == 2) {
    auto sizes = musa_ep::ReadIntInitializerNoLimit(split_inputs[1]);
    if (!sizes.has_value() || sizes->size() != split_outputs.size() ||
        sizes->empty() || (*sizes)[0] <= 0) {
      throw std::runtime_error("SplitConcat requires constant Split sizes");
    }
    part_width = (*sizes)[0];
  }
  std::string source_name = Name(split_inputs[0]);
  if (reshape_node) {
    std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
    std::vector<Ort::ConstValueInfo> reshape_outputs =
        reshape_node.GetOutputs();
    auto reshape_shape = GetTensorShape(reshape_outputs[0]);
    if (reshape_inputs.empty() || reshape_outputs.size() != 1 ||
        Name(reshape_outputs[0]) != source_name || !reshape_shape.has_value() ||
        reshape_shape->size() != 3 || (*reshape_shape)[1] <= 0 ||
        (*reshape_shape)[2] <= 0 || (*reshape_shape)[2] % part_count != 0) {
      throw std::runtime_error("SplitConcat invalid upstream Reshape");
    }
    source_name = Name(reshape_inputs[0]);
    sequence = (*reshape_shape)[1];
    const int64_t reshaped_part_width = (*reshape_shape)[2] / part_count;
    if (part_width > 0 && part_width != reshaped_part_width) {
      throw std::runtime_error("SplitConcat Reshape and Split sizes disagree");
    }
    part_width = reshaped_part_width;
  }

  bool transpose_output = false;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Transpose")) {
      auto perm = musa_ep::GetIntsAttribute(node, "perm");
      if (!perm.has_value() || *perm != std::vector<int64_t>{0, 2, 1}) {
        throw std::runtime_error("SplitConcat only supports perm=[0,2,1]");
      }
      transpose_output = true;
    }
  }

  std::unordered_map<std::string, size_t> fused_input_indices =
      FusedInputIndices(fused_node);
  return std::make_unique<SplitConcatFusionCompute>(
      GetFusedInputIndex(fused_input_indices, source_name), part_count,
      transpose_output, sequence, part_width);
}
