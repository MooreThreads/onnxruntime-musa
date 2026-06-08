// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/shape_cast_source_fusion.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/shape_gather_impl.h"

namespace {

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(Ort::ConstNode node, const char* op_type) {
  return node && node.GetOperatorType() == op_type &&
         IsOnnxDomain(node.GetDomain());
}

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
}

int64_t ReadIntAttribute(Ort::ConstNode node, const std::string& name,
                         int64_t default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK()) {
    return default_value;
  }

  int64_t value = default_value;
  status = attr.GetValue(value);
  return status.IsOK() ? value : default_value;
}

std::vector<int64_t> ReadIntInitializer(Ort::ConstValueInfo value_info) {
  if (!value_info || !value_info.IsConstantInitializer()) {
    throw std::runtime_error("ShapeCastSource requires int initializer");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("ShapeCastSource failed to read initializer");
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return ReadTyped<int64_t>(value);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> vals = ReadTyped<int32_t>(value);
    return std::vector<int64_t>(vals.begin(), vals.end());
  }
  throw std::runtime_error("ShapeCastSource constants must be int32/int64");
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
    throw std::runtime_error("unable to map ShapeCastSource input " +
                             input_name);
  }
  return it->second;
}

std::unordered_map<std::string, size_t> FusedOutputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> fused_output_indices;
  std::vector<Ort::ConstValueInfo> fused_outputs = fused_node.GetOutputs();
  for (size_t i = 0; i < fused_outputs.size(); ++i) {
    fused_output_indices.emplace(Name(fused_outputs[i]), i);
  }
  return fused_output_indices;
}

std::optional<size_t> FindFusedOutputIndex(
    const std::unordered_map<std::string, size_t>& fused_output_indices,
    const std::string& output_name) {
  auto it = fused_output_indices.find(output_name);
  if (it == fused_output_indices.end()) {
    return std::nullopt;
  }
  return it->second;
}

size_t DimIndexFromSplitOutput(Ort::ConstNode split_node,
                               const std::string& output_name) {
  if (ReadIntAttribute(split_node, "axis", 0) != 0) {
    throw std::runtime_error("ShapeCastSource Split must use axis 0");
  }
  std::vector<Ort::ConstValueInfo> inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> outputs = split_node.GetOutputs();
  if (inputs.empty() || outputs.empty()) {
    throw std::runtime_error("ShapeCastSource invalid Split");
  }

  int64_t input_count = inputs[0].TypeInfo()
                            .GetTensorTypeAndShapeInfo()
                            .GetElementCount();
  if (input_count < 0) {
    input_count = static_cast<int64_t>(outputs.size());
  }

  std::vector<int64_t> split_sizes;
  if (inputs.size() > 1 && inputs[1]) {
    split_sizes = ReadIntInitializer(inputs[1]);
  } else {
    if (input_count % static_cast<int64_t>(outputs.size()) != 0) {
      throw std::runtime_error("ShapeCastSource uneven Split");
    }
    split_sizes.assign(outputs.size(),
                       input_count / static_cast<int64_t>(outputs.size()));
  }
  if (split_sizes.size() != outputs.size()) {
    throw std::runtime_error("ShapeCastSource Split size mismatch");
  }

  int64_t offset = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (Name(outputs[i]) == output_name) {
      if (split_sizes[i] != 1) {
        throw std::runtime_error("ShapeCastSource Split output must be scalar");
      }
      return static_cast<size_t>(offset);
    }
    offset += split_sizes[i];
  }
  throw std::runtime_error("ShapeCastSource Split output not found");
}

size_t DimIndexFromSliceOutput(Ort::ConstNode slice_node) {
  std::vector<Ort::ConstValueInfo> inputs = slice_node.GetInputs();
  if (inputs.size() < 3 || inputs.size() > 5) {
    throw std::runtime_error("ShapeCastSource invalid Slice");
  }
  std::vector<int64_t> starts = ReadIntInitializer(inputs[1]);
  std::vector<int64_t> ends = ReadIntInitializer(inputs[2]);
  if (starts.size() != 1 || ends.size() != 1) {
    throw std::runtime_error("ShapeCastSource Slice must select one dim");
  }

  std::vector<int64_t> axes = {0};
  if (inputs.size() > 3 && inputs[3]) {
    axes = ReadIntInitializer(inputs[3]);
  }
  std::vector<int64_t> steps = {1};
  if (inputs.size() > 4 && inputs[4]) {
    steps = ReadIntInitializer(inputs[4]);
  }
  if (axes.size() != 1 || steps.size() != 1 || axes[0] != 0 ||
      steps[0] != 1 || ends[0] - starts[0] != 1 || starts[0] < 0) {
    throw std::runtime_error("ShapeCastSource unsupported Slice");
  }
  return static_cast<size_t>(starts[0]);
}

Ort::ConstNode FindShapeNode(Ort::ConstGraph graph) {
  Ort::ConstNode shape_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, "Shape")) {
      continue;
    }
    if (shape_node) {
      throw std::runtime_error("ShapeCastSource expects one Shape node");
    }
    shape_node = node;
  }
  if (!shape_node) {
    throw std::runtime_error("ShapeCastSource expects a Shape node");
  }
  return shape_node;
}

Ort::ConstNode FindShapeCastNode(Ort::ConstGraph graph) {
  Ort::ConstNode cast_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, "Cast")) {
      continue;
    }
    if (cast_node) {
      throw std::runtime_error("ShapeCastSource expects one Cast node");
    }
    cast_node = node;
  }
  if (!cast_node) {
    throw std::runtime_error("ShapeCastSource expects a Cast node");
  }
  return cast_node;
}

template <typename T>
OrtStatus* WriteOutput(Ort::UnownedValue output,
                       const std::vector<int64_t>& values) {
  std::vector<T> typed;
  typed.reserve(values.size());
  for (int64_t value : values) {
    if constexpr (std::is_same_v<T, int32_t>) {
      if (value > std::numeric_limits<int32_t>::max() ||
          value < std::numeric_limits<int32_t>::min()) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "ShapeCastSource int32 overflow");
      }
    }
    typed.push_back(static_cast<T>(value));
  }
  return WriteTyped<T>(output, typed);
}

OrtStatus* WriteShapeOutput(Ort::UnownedValue output,
                            const std::vector<int64_t>& values,
                            int64_t output_type) {
  if (IsGpuMemory(output.GetTensorMemoryInfo()) && values.size() <= 8) {
    MusaShapeGatherParams params{};
    params.output_count = static_cast<int32_t>(values.size());
    params.rank = static_cast<int32_t>(values.size());
    params.output_type = static_cast<int32_t>(output_type);
    for (size_t i = 0; i < values.size(); ++i) {
      params.dims[i] = values[i];
      params.indices[i] = static_cast<int64_t>(i);
    }
    return LaunchStatus(LaunchMusaShapeGatherKernel(
        output.GetTensorMutableRawData(), params, nullptr));
  }

  if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return WriteOutput<int32_t>(output, values);
  }
  return WriteOutput<int64_t>(output, values);
}

std::vector<ShapeCastSourceOutputPlan> BuildOutputPlans(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  auto fused_output_indices = FusedOutputIndices(fused_node);
  Ort::ConstNode cast_node = FindShapeCastNode(graph);
  const int64_t output_type = ReadIntAttribute(cast_node, "to", 0);
  if (output_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
      output_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::runtime_error("ShapeCastSource Cast must output int32/int64");
  }

  std::vector<ShapeCastSourceOutputPlan> outputs;
  std::vector<Ort::ConstValueInfo> cast_outputs = cast_node.GetOutputs();
  if (cast_outputs.size() != 1) {
    throw std::runtime_error("ShapeCastSource invalid Cast output");
  }
  if (auto output_index =
          FindFusedOutputIndex(fused_output_indices, Name(cast_outputs[0]));
      output_index.has_value()) {
    ShapeCastSourceOutputPlan plan;
    plan.output_index = *output_index;
    plan.output_type = output_type;
    plan.full_data_shape = true;
    outputs.push_back(plan);
  }

  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, "Slice") && !IsOnnxOp(node, "Split")) {
      continue;
    }
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      auto output_index =
          FindFusedOutputIndex(fused_output_indices, Name(output));
      if (!output_index.has_value()) {
        continue;
      }
      ShapeCastSourceOutputPlan plan;
      plan.output_index = *output_index;
      plan.output_type = output_type;
      plan.dim_index = IsOnnxOp(node, "Split")
                           ? DimIndexFromSplitOutput(node, Name(output))
                           : DimIndexFromSliceOutput(node);
      outputs.push_back(plan);
    }
  }

  if (outputs.empty()) {
    throw std::runtime_error("ShapeCastSource requires external outputs");
  }
  return outputs;
}

}  // namespace

ShapeCastSourceFusionCompute::ShapeCastSourceFusionCompute(
    size_t data_input_index, std::vector<ShapeCastSourceOutputPlan> outputs)
    : data_input_index(data_input_index), outputs(std::move(outputs)) {}

OrtStatus* ShapeCastSourceFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    Ort::ConstValue data = ctx.GetInput(data_input_index);
    std::vector<int64_t> data_shape =
        data.GetTensorTypeAndShapeInfo().GetShape();

    for (const ShapeCastSourceOutputPlan& plan : outputs) {
      std::vector<int64_t> values;
      if (plan.full_data_shape) {
        values = data_shape;
      } else {
        if (plan.dim_index >= data_shape.size()) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT, "ShapeCastSource dim index out of range");
        }
        values.push_back(data_shape[plan.dim_index]);
      }

      Ort::UnownedValue output =
          ctx.GetOutput(plan.output_index,
                        {static_cast<int64_t>(values.size())});
      OrtStatus* status = WriteShapeOutput(output, values, plan.output_type);
      if (status != nullptr) {
        return status;
      }
    }
    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

bool IsShapeCastSourceFusionGraph(Ort::ConstGraph graph) {
  bool has_shape = false;
  bool has_cast = false;
  bool has_branch = false;
  bool has_payload_consumer = false;
  for (Ort::ConstNode node : graph.GetNodes()) {
    has_shape = has_shape || IsOnnxOp(node, "Shape");
    has_cast = has_cast || IsOnnxOp(node, "Cast");
    has_branch =
        has_branch || IsOnnxOp(node, "Slice") || IsOnnxOp(node, "Split");
    has_payload_consumer =
        has_payload_consumer || IsOnnxOp(node, "Concat") ||
        IsOnnxOp(node, "Reshape") || IsOnnxOp(node, "Transpose");
  }
  return has_shape && has_cast && has_branch && !has_payload_consumer;
}

std::unique_ptr<FusionNodeCompute> CreateShapeCastSourceFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode shape_node = FindShapeNode(graph);
  auto fused_input_indices = FusedInputIndices(fused_node);
  std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
  if (shape_inputs.size() != 1) {
    throw std::runtime_error("ShapeCastSource invalid Shape node");
  }

  return std::make_unique<ShapeCastSourceFusionCompute>(
      GetFusedInputIndex(fused_input_indices, Name(shape_inputs[0])),
      BuildOutputPlans(graph, fused_node));
}
