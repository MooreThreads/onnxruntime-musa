// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/shape_cast_split_fusion.h"

#include <algorithm>
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
    throw std::runtime_error("ShapeCastSplit requires int initializer");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("ShapeCastSplit failed to read initializer");
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
  throw std::runtime_error("ShapeCastSplit constants must be int32/int64");
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
    throw std::runtime_error("unable to map ShapeCastSplit input " +
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
    throw std::runtime_error("unable to map ShapeCastSplit output " +
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
        throw std::runtime_error("ShapeCastSplit zero dim exceeds input rank");
      }
      out_shape[i] = input_shape[i];
    }
    if (out_shape[i] == -1) {
      if (infer_idx >= 0) {
        throw std::runtime_error("ShapeCastSplit only supports one inferred dim");
      }
      infer_idx = static_cast<int64_t>(i);
    } else {
      known *= out_shape[i];
    }
  }
  if (infer_idx >= 0) {
    if (known == 0 || input_size % known != 0) {
      throw std::runtime_error("ShapeCastSplit cannot infer output dim");
    }
    out_shape[static_cast<size_t>(infer_idx)] = input_size / known;
  }
  if (NumElements(out_shape) != input_size) {
    throw std::runtime_error("ShapeCastSplit element count mismatch");
  }
  return out_shape;
}

std::vector<ShapeCastSplitTerm> BuildShapeTerms(Ort::ConstNode concat_node) {
  std::vector<ShapeCastSplitTerm> terms;
  int64_t dynamic_term_count = 0;
  int64_t infer_count = 0;
  for (Ort::ConstValueInfo input : concat_node.GetInputs()) {
    if (!input.IsConstantInitializer()) {
      ++dynamic_term_count;
      terms.push_back(ShapeCastSplitTerm{true, {}});
      continue;
    }

    ShapeCastSplitTerm term;
    term.values = ReadIntInitializer(input);
    infer_count += static_cast<int64_t>(
        std::count(term.values.begin(), term.values.end(), -1));
    terms.push_back(std::move(term));
  }
  if (terms.empty() || dynamic_term_count > 1 || infer_count > 1 ||
      dynamic_term_count == 0) {
    throw std::runtime_error(
        "ShapeCastSplit requires one dynamic batch shape term");
  }
  return terms;
}

std::vector<int64_t> ResolveShapeTerms(
    const std::vector<ShapeCastSplitTerm>& terms,
    const std::vector<int64_t>& data_shape) {
  if (data_shape.empty()) {
    throw std::runtime_error("ShapeCastSplit requires ranked data input");
  }
  std::vector<int64_t> values;
  for (const ShapeCastSplitTerm& term : terms) {
    if (term.from_data_dim0) {
      values.push_back(data_shape[0]);
    } else {
      values.insert(values.end(), term.values.begin(), term.values.end());
    }
  }
  return values;
}

int64_t Product(const std::vector<int64_t>& shape, size_t begin, size_t end) {
  int64_t value = 1;
  for (size_t i = begin; i < end; ++i) {
    value *= shape[i];
  }
  return value;
}

Ort::ConstNode FindSingleNode(Ort::ConstGraph graph, const char* op_type) {
  Ort::ConstNode found{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, op_type)) {
      continue;
    }
    if (found) {
      throw std::runtime_error(std::string("ShapeCastSplit expects one ") +
                               op_type + " node");
    }
    found = node;
  }
  if (!found) {
    throw std::runtime_error(std::string("ShapeCastSplit expects a ") +
                             op_type + " node");
  }
  return found;
}

std::vector<int64_t> SplitSizes(Ort::ConstNode split_node,
                                int64_t axis_dim) {
  std::vector<Ort::ConstValueInfo> inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> outputs = split_node.GetOutputs();
  if (outputs.empty()) {
    throw std::runtime_error("ShapeCastSplit requires Split outputs");
  }
  if (inputs.size() > 1 && inputs[1]) {
    return ReadIntInitializer(inputs[1]);
  }
  if (axis_dim % static_cast<int64_t>(outputs.size()) != 0) {
    throw std::runtime_error("ShapeCastSplit requires even Split sizes");
  }
  return std::vector<int64_t>(
      outputs.size(), axis_dim / static_cast<int64_t>(outputs.size()));
}

}  // namespace

ShapeCastSplitFusionCompute::ShapeCastSplitFusionCompute(
    size_t data_input_index, std::vector<ShapeCastSplitTerm> shape_terms,
    int64_t split_axis, std::vector<ShapeCastSplitOutputPlan> outputs,
    int64_t allowzero)
    : data_input_index(data_input_index),
      shape_terms(std::move(shape_terms)),
      split_axis(split_axis),
      outputs(std::move(outputs)),
      allowzero(allowzero) {}

OrtStatus* ShapeCastSplitFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    Ort::ConstValue data = ctx.GetInput(data_input_index);
    auto data_info = data.GetTensorTypeAndShapeInfo();
    auto elem_type = data_info.GetElementType();
    std::vector<int64_t> data_shape = data_info.GetShape();
    std::vector<int64_t> reshape_shape = ResolveReshapeOutputShape(
        data_shape, ResolveShapeTerms(shape_terms, data_shape), allowzero);
    int64_t axis = NormalizeAxis(split_axis, reshape_shape.size());
    const size_t elem_size = ElementSize(elem_type);
    if (elem_size == 0) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "ShapeCastSplit unsupported dtype");
    }
    if (!IsGpuMemory(data.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "ShapeCastSplit requires MUSA input");
    }

    const int64_t outer =
        Product(reshape_shape, 0, static_cast<size_t>(axis));
    const int64_t inner =
        Product(reshape_shape, static_cast<size_t>(axis) + 1,
                reshape_shape.size());
    const int64_t axis_dim = reshape_shape[static_cast<size_t>(axis)];
    const size_t src_pitch =
        static_cast<size_t>(axis_dim * inner) * elem_size;
    const auto* src_base = static_cast<const uint8_t*>(data.GetTensorRawData());

    for (const ShapeCastSplitOutputPlan& plan : outputs) {
      int64_t axis_width = plan.axis_width;
      int64_t axis_offset = plan.axis_offset;
      if (axis_width == 0) {
        if (outputs.empty() ||
            axis_dim % static_cast<int64_t>(outputs.size()) != 0) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT, "ShapeCastSplit uneven runtime Split");
        }
        axis_width = axis_dim / static_cast<int64_t>(outputs.size());
        axis_offset *= axis_width;
      }
      if (axis_width <= 0 || axis_offset < 0 ||
          axis_offset + axis_width > axis_dim) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "ShapeCastSplit invalid Split range");
      }

      std::vector<int64_t> output_shape = reshape_shape;
      output_shape[static_cast<size_t>(axis)] = axis_width;
      Ort::UnownedValue output = ctx.GetOutput(plan.output_index, output_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "ShapeCastSplit requires MUSA output");
      }

      const size_t width_bytes =
          static_cast<size_t>(axis_width * inner) * elem_size;
      const size_t dst_pitch = width_bytes;
      const size_t src_offset =
          static_cast<size_t>(axis_offset * inner) * elem_size;
      OrtStatus* status =
          DeviceMemcpy2D(output.GetTensorMutableRawData(), dst_pitch,
                         src_base + src_offset, src_pitch, width_bytes,
                         static_cast<size_t>(outer));
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

bool IsShapeCastSplitFusionGraph(Ort::ConstGraph graph) {
  size_t concat_count = 0;
  size_t cast_count = 0;
  size_t reshape_count = 0;
  size_t split_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    concat_count += IsOnnxOp(node, "Concat") ? 1 : 0;
    cast_count += IsOnnxOp(node, "Cast") ? 1 : 0;
    reshape_count += IsOnnxOp(node, "Reshape") ? 1 : 0;
    split_count += IsOnnxOp(node, "Split") ? 1 : 0;
  }
  return concat_count == 1 && cast_count == 1 && reshape_count == 1 &&
         split_count == 1;
}

std::unique_ptr<FusionNodeCompute> CreateShapeCastSplitFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode concat_node = FindSingleNode(graph, "Concat");
  Ort::ConstNode reshape_node = FindSingleNode(graph, "Reshape");
  Ort::ConstNode split_node = FindSingleNode(graph, "Split");

  auto fused_input_indices = FusedInputIndices(fused_node);
  auto fused_output_indices = FusedOutputIndices(fused_node);
  std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
  if (reshape_inputs.size() != 2) {
    throw std::runtime_error("ShapeCastSplit invalid Reshape node");
  }

  std::vector<ShapeCastSplitTerm> shape_terms = BuildShapeTerms(concat_node);
  int64_t split_axis = ReadIntAttribute(split_node, "axis", 0);
  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  std::vector<int64_t> split_sizes;
  if (split_inputs.size() > 1 && split_inputs[1]) {
    split_sizes = ReadIntInitializer(split_inputs[1]);
    if (split_sizes.size() != split_outputs.size()) {
      throw std::runtime_error("ShapeCastSplit Split size mismatch");
    }
  }

  int64_t offset = 0;
  std::vector<ShapeCastSplitOutputPlan> outputs;
  for (size_t i = 0; i < split_outputs.size(); ++i) {
    const int64_t width = split_sizes.empty() ? 0 : split_sizes[i];
    outputs.push_back(ShapeCastSplitOutputPlan{
        GetFusedOutputIndex(fused_output_indices, Name(split_outputs[i])),
        split_sizes.empty() ? static_cast<int64_t>(i) : offset, width});
    offset += width;
  }

  return std::make_unique<ShapeCastSplitFusionCompute>(
      GetFusedInputIndex(fused_input_indices, Name(reshape_inputs[0])),
      std::move(shape_terms), split_axis, std::move(outputs),
      ReadIntAttribute(reshape_node, "allowzero", 0));
}
