// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/shape_gather_fusion.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/shape_gather_impl.h"

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

bool IsSupportedOutputType(int64_t elem_type) {
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

std::vector<Ort::ConstNode> Nodes(Ort::ConstGraph graph) {
  return graph.GetNodes();
}

bool FindShapeCastGatherNodes(Ort::ConstGraph graph, Ort::ConstNode& shape_node,
                              Ort::ConstNode& cast_node,
                              Ort::ConstNode& gather_node) {
  for (Ort::ConstNode node : Nodes(graph)) {
    if (IsOnnxOp(node, "Shape")) {
      shape_node = node;
    } else if (IsOnnxOp(node, "Cast")) {
      cast_node = node;
    } else if (IsOnnxOp(node, "Gather")) {
      gather_node = node;
    }
  }
  return shape_node && cast_node && gather_node;
}

bool ValidateShapeCastGather(Ort::ConstNode shape_node,
                             Ort::ConstNode cast_node,
                             Ort::ConstNode gather_node) {
  std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> shape_outputs = shape_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> cast_inputs = cast_node.GetInputs();
  std::vector<Ort::ConstValueInfo> cast_outputs = cast_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gather_outputs = gather_node.GetOutputs();
  if (shape_inputs.size() != 1 || shape_outputs.size() != 1 ||
      cast_inputs.size() != 1 || cast_outputs.size() != 1 ||
      gather_inputs.size() != 2 || gather_outputs.size() != 1) {
    return false;
  }
  if (Name(shape_outputs[0]) != Name(cast_inputs[0]) ||
      Name(cast_outputs[0]) != Name(gather_inputs[0])) {
    return false;
  }
  if (ReadIntAttribute(gather_node, "axis", 0) != 0) {
    return false;
  }
  return IsSupportedOutputType(ReadIntAttribute(cast_node, "to", 0));
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
    throw std::runtime_error("unable to map ShapeCastGather input " +
                             input_name);
  }
  return it->second;
}

std::vector<int64_t> ReadIndices(Ort::ConstValue value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return ReadTyped<int64_t>(value);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> values = ReadTyped<int32_t>(value);
    return std::vector<int64_t>(values.begin(), values.end());
  }
  throw std::runtime_error("ShapeCastGather requires int32/int64 indices");
}

std::vector<int64_t> ReadInitializerIndices(Ort::ConstValueInfo value_info) {
  if (!value_info.IsConstantInitializer()) {
    return {};
  }
  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    return {};
  }
  return ReadIndices(value);
}

template <typename T>
OrtStatus* WriteCpuOutput(Ort::UnownedValue output,
                          const std::vector<int64_t>& values) {
  std::vector<T> out;
  out.reserve(values.size());
  for (int64_t value : values) {
    if constexpr (std::is_same_v<T, int32_t>) {
      if (value > std::numeric_limits<int32_t>::max() ||
          value < std::numeric_limits<int32_t>::min()) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "ShapeCastGather int32 overflow");
      }
    }
    out.push_back(static_cast<T>(value));
  }
  return WriteTyped<T>(output, out);
}

}  // namespace

ShapeCastGatherFusionCompute::ShapeCastGatherFusionCompute(
    size_t data_input_index, size_t indices_input_index,
    std::vector<int64_t> cached_indices, int64_t output_type)
    : data_input_index(data_input_index),
      indices_input_index(indices_input_index),
      cached_indices(std::move(cached_indices)),
      output_type(output_type) {}

OrtStatus* ShapeCastGatherFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    Ort::ConstValue data = ctx.GetInput(data_input_index);
    std::vector<int64_t> data_shape =
        data.GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> indices_shape;
    std::vector<int64_t> indices;
    if (!cached_indices.empty()) {
      indices = cached_indices;
      indices_shape = {static_cast<int64_t>(indices.size())};
    } else {
      Ort::ConstValue indices_value = ctx.GetInput(indices_input_index);
      indices_shape = indices_value.GetTensorTypeAndShapeInfo().GetShape();
      indices = ReadIndices(indices_value);
    }
    if (data_shape.size() > 8 || indices.size() > 8) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "ShapeCastGather rank exceeds MUSA limit");
    }

    std::vector<int64_t> values;
    values.reserve(indices.size());
    for (int64_t index : indices) {
      if (index < 0) {
        index += static_cast<int64_t>(data_shape.size());
      }
      if (index < 0 || index >= static_cast<int64_t>(data_shape.size())) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "ShapeCastGather index out of range");
      }
      values.push_back(data_shape[static_cast<size_t>(index)]);
    }

    Ort::UnownedValue y = ctx.GetOutput(0, indices_shape);
    if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
      if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        return WriteCpuOutput<int32_t>(y, values);
      }
      return WriteCpuOutput<int64_t>(y, values);
    }

    MusaShapeGatherParams params{};
    params.output_count = static_cast<int32_t>(indices.size());
    params.rank = static_cast<int32_t>(data_shape.size());
    params.output_type = static_cast<int32_t>(output_type);
    for (size_t i = 0; i < data_shape.size(); ++i) {
      params.dims[i] = data_shape[i];
    }
    for (size_t i = 0; i < indices.size(); ++i) {
      params.indices[i] = indices[i];
    }
    return LaunchStatus(LaunchMusaShapeGatherKernel(y.GetTensorMutableRawData(),
                                                    params, nullptr));
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

bool IsShapeCastGatherFusionGraph(Ort::ConstGraph graph) {
  std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  if (nodes.size() != 3) {
    return false;
  }
  Ort::ConstNode shape_node{nullptr};
  Ort::ConstNode cast_node{nullptr};
  Ort::ConstNode gather_node{nullptr};
  return FindShapeCastGatherNodes(graph, shape_node, cast_node, gather_node) &&
         ValidateShapeCastGather(shape_node, cast_node, gather_node);
}

std::unique_ptr<FusionNodeCompute> CreateShapeCastGatherFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode shape_node{nullptr};
  Ort::ConstNode cast_node{nullptr};
  Ort::ConstNode gather_node{nullptr};
  if (!FindShapeCastGatherNodes(graph, shape_node, cast_node, gather_node) ||
      !ValidateShapeCastGather(shape_node, cast_node, gather_node)) {
    throw std::runtime_error(
        "ShapeCastGather fusion expects Shape+Cast+Gather");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  std::vector<int64_t> cached_indices =
      ReadInitializerIndices(gather_inputs[1]);
  size_t indices_input_index = 0;
  if (cached_indices.empty()) {
    indices_input_index =
        GetFusedInputIndex(fused_input_indices, Name(gather_inputs[1]));
  }
  return std::make_unique<ShapeCastGatherFusionCompute>(
      GetFusedInputIndex(fused_input_indices, Name(shape_inputs[0])),
      indices_input_index, std::move(cached_indices),
      ReadIntAttribute(cast_node, "to", 0));
}
