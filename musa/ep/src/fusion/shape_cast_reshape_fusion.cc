// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/shape_cast_reshape_fusion.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "kernels/shared_inc/op_kernel_common.h"

namespace {

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(Ort::ConstNode node, const char* op_type) {
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

std::vector<int64_t> ReadIntInitializer(Ort::ConstValueInfo value_info) {
  if (!value_info || !value_info.IsConstantInitializer()) {
    throw std::runtime_error("ShapeCastReshape requires int initializer");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("ShapeCastReshape failed to read initializer");
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
  throw std::runtime_error("ShapeCastReshape constants must be int32/int64");
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
    throw std::runtime_error("unable to map ShapeCastReshape input " +
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

size_t GetFusedOutputIndex(
    const std::unordered_map<std::string, size_t>& fused_output_indices,
    const std::string& output_name) {
  auto it = fused_output_indices.find(output_name);
  if (it == fused_output_indices.end()) {
    throw std::runtime_error("unable to map ShapeCastReshape output " +
                             output_name);
  }
  return it->second;
}

Ort::ConstNode ProducerOf(Ort::ConstValueInfo value_info) {
  Ort::ValueInfoConsumerProducerInfo producer = value_info.GetProducerNode();
  return producer.node;
}

std::vector<int64_t> ResolveReshapeOutputShape(
    const std::vector<int64_t>& input_shape, std::vector<int64_t> out_shape,
    int64_t allowzero) {
  int64_t input_size = NumElements(input_shape);
  int64_t known = 1;
  int64_t infer_idx = -1;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    if (out_shape[i] == 0 && !allowzero) {
      if (i >= input_shape.size()) {
        throw std::runtime_error(
            "ShapeCastReshape zero dim exceeds input rank");
      }
      out_shape[i] = input_shape[i];
    }
    if (out_shape[i] == -1) {
      if (infer_idx >= 0) {
        throw std::runtime_error(
            "ShapeCastReshape only supports one inferred dim");
      }
      infer_idx = static_cast<int64_t>(i);
    } else {
      known *= out_shape[i];
    }
  }
  if (infer_idx >= 0) {
    if (known == 0 || input_size % known != 0) {
      throw std::runtime_error("ShapeCastReshape cannot infer output dim");
    }
    out_shape[static_cast<size_t>(infer_idx)] = input_size / known;
  }
  return out_shape;
}

Ort::ConstNode FindConcatNode(Ort::ConstGraph graph) {
  Ort::ConstNode concat_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, "Concat")) {
      continue;
    }
    if (concat_node) {
      throw std::runtime_error("ShapeCastReshape expects one Concat node");
    }
    concat_node = node;
  }
  if (!concat_node) {
    throw std::runtime_error("ShapeCastReshape expects a Concat node");
  }
  return concat_node;
}

Ort::ConstNode FindCastNode(Ort::ConstGraph graph) {
  Ort::ConstNode cast_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, "Cast")) {
      continue;
    }
    if (cast_node) {
      throw std::runtime_error("ShapeCastReshape expects one Cast node");
    }
    cast_node = node;
  }
  if (!cast_node) {
    throw std::runtime_error("ShapeCastReshape expects a Cast node");
  }
  return cast_node;
}

std::vector<ShapeCastReshapeTerm> BuildRequestedShapeTerms(
    Ort::ConstNode concat_node,
    const std::unordered_map<std::string, size_t>& fused_input_indices) {
  std::vector<ShapeCastReshapeTerm> requested_shape_terms;
  int64_t constant_infer_count = 0;
  for (Ort::ConstValueInfo input : concat_node.GetInputs()) {
    if (!input.IsConstantInitializer()) {
      continue;
    }
    std::vector<int64_t> values = ReadIntInitializer(input);
    constant_infer_count +=
        static_cast<int64_t>(std::count(values.begin(), values.end(), -1));
  }
  if (constant_infer_count > 1) {
    throw std::runtime_error(
        "ShapeCastReshape requires at most one inferred constant dim");
  }

  int64_t dynamic_term_count = 0;
  for (Ort::ConstValueInfo input : concat_node.GetInputs()) {
    if (input.IsConstantInitializer()) {
      std::vector<int64_t> values = ReadIntInitializer(input);
      for (int64_t value : values) {
        requested_shape_terms.push_back(
          ShapeCastReshapeTerm{ShapeCastReshapeTerm::Kind::kConstant, value,
                                 0});
      }
      continue;
    }

    ++dynamic_term_count;
    requested_shape_terms.push_back(
        ShapeCastReshapeTerm{ShapeCastReshapeTerm::Kind::kInputScalar, 0,
                             GetFusedInputIndex(fused_input_indices,
                                                Name(input))});
  }

  if (requested_shape_terms.empty() || dynamic_term_count > 1 ||
      (constant_infer_count == 0 && dynamic_term_count == 0)) {
    throw std::runtime_error(
        "ShapeCastReshape requires one dynamic shape term at most");
  }
  return requested_shape_terms;
}

std::vector<ShapeCastReshapePlan> BuildOutputPlans(Ort::ConstGraph graph,
                                                   Ort::ConstNode fused_node) {
  Ort::ConstNode concat_node = FindConcatNode(graph);
  auto fused_input_indices = FusedInputIndices(fused_node);
  auto fused_output_indices = FusedOutputIndices(fused_node);
  std::vector<ShapeCastReshapeTerm> requested_shape_terms =
      BuildRequestedShapeTerms(concat_node, fused_input_indices);

  std::vector<ShapeCastReshapePlan> outputs;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, "Reshape")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> node_outputs = node.GetOutputs();
    if (inputs.size() != 2 || node_outputs.size() != 1) {
      throw std::runtime_error("ShapeCastReshape invalid Reshape node");
    }
    ShapeCastReshapePlan plan;
    plan.data_input_index =
        GetFusedInputIndex(fused_input_indices, Name(inputs[0]));
    plan.output_index =
        GetFusedOutputIndex(fused_output_indices, Name(node_outputs[0]));
    plan.requested_shape_terms = requested_shape_terms;
    plan.allowzero = ReadIntAttribute(node, "allowzero", 0);
    outputs.push_back(std::move(plan));
  }
  if (outputs.empty()) {
    throw std::runtime_error("ShapeCastReshape requires Reshape outputs");
  }
  return outputs;
}

}  // namespace

ShapeCastReshapeFusionCompute::ShapeCastReshapeFusionCompute(
    std::vector<ShapeCastReshapePlan> outputs)
    : outputs(std::move(outputs)) {}

int64_t ReadInputScalar(Ort::ConstValue value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementCount() != 1) {
    throw std::runtime_error("ShapeCastReshape dynamic shape term must be scalar");
  }

  std::vector<uint8_t> bytes;
  const void* raw_data = value.GetTensorRawData();
  if (IsGpuMemory(value.GetTensorMemoryInfo())) {
    Ort::ThrowOnError(CopyToHost(value, bytes));
    raw_data = bytes.data();
  }

  ONNXTensorElementDataType elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    int32_t value_i32 = 0;
    std::memcpy(&value_i32, raw_data, sizeof(value_i32));
    return static_cast<int64_t>(value_i32);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    int64_t value_i64 = 0;
    std::memcpy(&value_i64, raw_data, sizeof(value_i64));
    return value_i64;
  }
  throw std::runtime_error(
      "ShapeCastReshape dynamic shape term must be int32/int64");
}

std::vector<int64_t> ResolveRequestedShapeTerms(
    Ort::KernelContext& ctx,
    const std::vector<ShapeCastReshapeTerm>& terms) {
  std::vector<int64_t> requested_shape;
  requested_shape.reserve(terms.size());
  for (const ShapeCastReshapeTerm& term : terms) {
    if (term.kind == ShapeCastReshapeTerm::Kind::kConstant) {
      requested_shape.push_back(term.value);
      continue;
    }
    requested_shape.push_back(ReadInputScalar(ctx.GetInput(term.input_index)));
  }
  return requested_shape;
}

OrtStatus* ShapeCastReshapeFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    for (const ShapeCastReshapePlan& plan : outputs) {
      Ort::ConstValue data = ctx.GetInput(plan.data_input_index);
      auto data_info = data.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> data_shape = data_info.GetShape();
      std::vector<int64_t> requested_shape =
          ResolveRequestedShapeTerms(ctx, plan.requested_shape_terms);
      std::vector<int64_t> output_shape = ResolveReshapeOutputShape(
          data_shape, std::move(requested_shape), plan.allowzero);

      const size_t elem_size = ElementSize(data_info.GetElementType());
      if (elem_size == 0) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "ShapeCastReshape unsupported dtype");
      }
      if (!IsGpuMemory(data.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "ShapeCastReshape requires MUSA input");
      }

      Ort::UnownedValue output = ctx.GetOutput(plan.output_index, output_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "ShapeCastReshape requires MUSA output");
      }

      OrtStatus* status = DeviceMemcpy(output.GetTensorMutableRawData(),
                                       data.GetTensorRawData(),
                                       data.GetTensorSizeInBytes());
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

bool IsShapeCastReshapeFusionGraph(Ort::ConstGraph graph) {
  bool has_concat = false;
  bool has_cast = false;
  bool has_reshape = false;
  bool has_shape = false;
  for (Ort::ConstNode node : graph.GetNodes()) {
    has_concat = has_concat || IsOnnxOp(node, "Concat");
    has_cast = has_cast || IsOnnxOp(node, "Cast");
    has_reshape = has_reshape || IsOnnxOp(node, "Reshape");
    has_shape = has_shape || IsOnnxOp(node, "Shape");
  }
  return has_concat && has_cast && has_reshape && !has_shape;
}

std::unique_ptr<FusionNodeCompute> CreateShapeCastReshapeFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode concat_node = FindConcatNode(graph);
  Ort::ConstNode cast_node = FindCastNode(graph);

  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> cast_inputs = cast_node.GetInputs();
  if (concat_outputs.size() != 1 || cast_inputs.size() != 1 ||
      Name(concat_outputs[0]) != Name(cast_inputs[0])) {
    throw std::runtime_error("ShapeCastReshape expects Concat feeding Cast");
  }
  if (ReadIntAttribute(cast_node, "to", 0) !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::runtime_error("ShapeCastReshape expects Cast to int64");
  }

  return std::make_unique<ShapeCastReshapeFusionCompute>(
      BuildOutputPlans(graph, fused_node));
}
