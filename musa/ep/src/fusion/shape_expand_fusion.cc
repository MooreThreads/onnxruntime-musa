// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/shape_expand_fusion.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "kernels/generator/constant_of_shape_impl.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/expand_impl.h"
#include "plugin_ep_utils.h"

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
    throw std::runtime_error("ShapeExpand requires constant shape input");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("ShapeExpand failed to read initializer");
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
  throw std::runtime_error("ShapeExpand constants must be int32/int64");
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

std::unordered_map<std::string, size_t> FusedOutputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> fused_output_indices;
  std::vector<Ort::ConstValueInfo> fused_outputs = fused_node.GetOutputs();
  for (size_t i = 0; i < fused_outputs.size(); ++i) {
    fused_output_indices.emplace(Name(fused_outputs[i]), i);
  }
  return fused_output_indices;
}

size_t GetMappedIndex(const std::unordered_map<std::string, size_t>& indices,
                      const std::string& name, const char* kind) {
  auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error(std::string("unable to map ShapeExpand ") + kind +
                             " " + name);
  }
  return it->second;
}

Ort::ConstNode ProducerOf(Ort::ConstValueInfo value_info) {
  Ort::ValueInfoConsumerProducerInfo producer = value_info.GetProducerNode();
  return producer.node;
}

Ort::ConstNode FindShapeNode(Ort::ConstGraph graph) {
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Shape")) {
      return node;
    }
  }
  return Ort::ConstNode{nullptr};
}

ShapeExpandTerm TermFromSliceOutput(
    Ort::ConstNode slice_node, const std::string& output_name,
    std::optional<size_t> shape_rank = std::nullopt) {
  std::vector<Ort::ConstValueInfo> slice_outputs = slice_node.GetOutputs();
  if (slice_outputs.size() != 1 || Name(slice_outputs[0]) != output_name) {
    throw std::runtime_error("ShapeExpand Slice output mismatch");
  }

  std::vector<Ort::ConstValueInfo> slice_inputs = slice_node.GetInputs();
  if (slice_inputs.size() < 3 || slice_inputs.size() > 5) {
    throw std::runtime_error("ShapeExpand unsupported Slice input count");
  }
  std::vector<int64_t> starts = ReadIntInitializer(slice_inputs[1]);
  std::vector<int64_t> ends = ReadIntInitializer(slice_inputs[2]);
  if (starts.size() != 1 || ends.size() != 1) {
    throw std::runtime_error("ShapeExpand Slice must select one range");
  }
  std::vector<int64_t> axes = {0};
  if (slice_inputs.size() > 3 && slice_inputs[3]) {
    axes = ReadIntInitializer(slice_inputs[3]);
  }
  std::vector<int64_t> steps = {1};
  if (slice_inputs.size() > 4 && slice_inputs[4]) {
    steps = ReadIntInitializer(slice_inputs[4]);
  }
  if (axes.size() != 1 || steps.size() != 1 || axes[0] != 0 ||
      steps[0] != 1) {
    throw std::runtime_error("ShapeExpand Slice must use axis 0 step 1");
  }
  int64_t start = starts[0];
  int64_t end = ends[0];
  if (shape_rank.has_value()) {
    const int64_t signed_shape_rank = static_cast<int64_t>(*shape_rank);
    start = start < 0 ? start + signed_shape_rank : start;
    end = end < 0 ? end + signed_shape_rank : end;
    start = std::max<int64_t>(0, std::min(start, signed_shape_rank));
    end = std::max<int64_t>(0, std::min(end, signed_shape_rank));
  } else if (start < 0 || end < 0) {
    throw std::runtime_error(
        "ShapeExpand dynamic-rank Slice requires non-negative bounds");
  }
  if (end <= start) {
    throw std::runtime_error("ShapeExpand Slice selects empty shape range");
  }
  return ShapeExpandTerm{true, start, end - start, {}};
}

std::vector<ShapeExpandTerm> BuildTerms(Ort::ConstNode concat_node,
                                        Ort::ConstNode shape_node) {
  std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
  if (shape_inputs.size() != 1) {
    throw std::runtime_error("ShapeExpand Shape input mismatch");
  }
  auto shape_source_shape = GetTensorShape(shape_inputs[0]);

  std::vector<ShapeExpandTerm> terms;
  for (Ort::ConstValueInfo input : concat_node.GetInputs()) {
    Ort::ConstNode producer = ProducerOf(input);
    if (producer && IsOnnxOp(producer, "Slice")) {
      terms.push_back(TermFromSliceOutput(
          producer, Name(input),
          shape_source_shape.has_value()
              ? std::optional<size_t>(shape_source_shape->size())
              : std::nullopt));
      continue;
    }
    ShapeExpandTerm term;
    term.values = ReadIntInitializer(input);
    terms.push_back(std::move(term));
  }
  return terms;
}

bool IsShapeCastOutput(Ort::ConstValueInfo value, Ort::ConstNode shape_node) {
  std::vector<Ort::ConstValueInfo> shape_outputs = shape_node.GetOutputs();
  if (shape_outputs.size() != 1) {
    return false;
  }
  for (const auto& consumer : shape_outputs[0].GetConsumers()) {
    if (consumer.index != 0 || !IsOnnxOp(consumer.node, "Cast")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> cast_outputs = consumer.node.GetOutputs();
    if (cast_outputs.size() == 1 && Name(cast_outputs[0]) == Name(value)) {
      return true;
    }
  }
  return false;
}

std::vector<ShapeExpandTerm> BuildTermsFromShapeValue(
    Ort::ConstValueInfo shape_value, Ort::ConstNode shape_node) {
  Ort::ConstNode producer = ProducerOf(shape_value);
  if (producer && IsOnnxOp(producer, "Concat") &&
      ReadIntAttribute(producer, "axis", 0) == 0) {
    return BuildTerms(producer, shape_node);
  }
  if (producer && IsOnnxOp(producer, "Slice")) {
    return {TermFromSliceOutput(producer, Name(shape_value))};
  }
  if (IsShapeCastOutput(shape_value, shape_node)) {
    return {ShapeExpandTerm{true, 0, -1, {}}};
  }
  throw std::runtime_error("ShapeExpand unsupported shape expression");
}

ShapeExpandOutputPlan::Kind OutputKind(Ort::ConstNode node) {
  if (IsOnnxOp(node, "Expand")) {
    return ShapeExpandOutputPlan::Kind::Expand;
  }
  if (IsOnnxOp(node, "ConstantOfShape")) {
    return ShapeExpandOutputPlan::Kind::ConstantOfShape;
  }
  throw std::runtime_error("ShapeExpand unsupported output op");
}

void InitConstantValue(Ort::ConstNode node, ShapeExpandOutputPlan& plan) {
  float default_value = 0.0f;
  plan.value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  plan.value_bits = 0;
  plan.value_size = sizeof(default_value);
  std::memcpy(&plan.value_bits, &default_value, sizeof(default_value));

  Ort::ConstOpAttr attr;
  Ort::Status attr_status = node.GetAttributeByName("value", attr);
  if (!attr_status.IsOK()) {
    return;
  }

  Ort::Value value{nullptr};
  Ort::Status status = attr.GetTensorAttributeAsOrtValue(value);
  if (!status.IsOK() || !value) {
    return;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementCount() == 0) {
    return;
  }
  if (info.GetElementCount() != 1) {
    throw std::runtime_error(
        "ShapeExpand ConstantOfShape value must be scalar");
  }
  plan.value_type = info.GetElementType();
  plan.value_size = ElementSize(plan.value_type);
  if (plan.value_size == 0 || plan.value_size > sizeof(uint64_t)) {
    throw std::runtime_error("ShapeExpand ConstantOfShape unsupported dtype");
  }
  plan.value_bits = 0;
  std::memcpy(&plan.value_bits, value.GetTensorRawData(), plan.value_size);
}

void InitScalarFillValue(Ort::ConstValueInfo value_info,
                         ShapeExpandOutputPlan& plan) {
  if (!value_info || !value_info.IsConstantInitializer()) {
    return;
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    return;
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementCount() != 1) {
    return;
  }
  const auto elem_type = info.GetElementType();
  const size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0 || elem_size > sizeof(uint64_t)) {
    return;
  }

  plan.has_scalar_fill_value = true;
  plan.scalar_fill_value_type = elem_type;
  plan.scalar_fill_value_size = elem_size;
  plan.scalar_fill_value_bits = 0;
  std::memcpy(&plan.scalar_fill_value_bits, value.GetTensorRawData(),
              elem_size);
}

std::vector<ShapeExpandOutputPlan> BuildOutputPlans(
    Ort::ConstGraph graph, Ort::ConstNode fused_node, Ort::ConstNode shape_node) {
  auto fused_input_indices = FusedInputIndices(fused_node);
  auto fused_output_indices = FusedOutputIndices(fused_node);
  std::vector<ShapeExpandOutputPlan> outputs;
  for (Ort::ConstNode node : graph.GetNodes()) {
    const bool is_expand = IsOnnxOp(node, "Expand");
    const bool is_constant_of_shape = IsOnnxOp(node, "ConstantOfShape");
    if (!is_expand && !is_constant_of_shape) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> node_inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> node_outputs = node.GetOutputs();
    if ((is_expand && node_inputs.size() != 2) ||
        (is_constant_of_shape && node_inputs.size() != 1) ||
        node_outputs.size() != 1) {
      continue;
    }
    Ort::ConstValueInfo shape_input = is_expand ? node_inputs[1] : node_inputs[0];
    Ort::ConstNode final_cast_node = ProducerOf(shape_input);
    if (is_constant_of_shape && final_cast_node &&
        final_cast_node.GetId() == shape_node.GetId()) {
      ShapeExpandOutputPlan plan;
      plan.kind = ShapeExpandOutputPlan::Kind::ConstantOfShape;
      plan.output_index =
          GetMappedIndex(fused_output_indices, Name(node_outputs[0]), "output");
      plan.terms = {ShapeExpandTerm{true, 0, -1, {}}};
      InitConstantValue(node, plan);
      outputs.push_back(std::move(plan));
      continue;
    }
    if (!final_cast_node || !IsOnnxOp(final_cast_node, "Cast") ||
        ReadIntAttribute(final_cast_node, "to", 0) !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> final_cast_inputs =
        final_cast_node.GetInputs();
    if (final_cast_inputs.size() != 1) {
      continue;
    }

    ShapeExpandOutputPlan plan;
    plan.kind = OutputKind(node);
    if (is_expand) {
      plan.data_input_index =
          GetMappedIndex(fused_input_indices, Name(node_inputs[0]), "input");
      InitScalarFillValue(node_inputs[0], plan);
    }
    plan.output_index =
        GetMappedIndex(fused_output_indices, Name(node_outputs[0]), "output");
    plan.terms = BuildTermsFromShapeValue(final_cast_inputs[0], shape_node);
    if (is_constant_of_shape) {
      InitConstantValue(node, plan);
    }
    outputs.push_back(std::move(plan));
  }
  return outputs;
}

}  // namespace

ShapeExpandFusionCompute::ShapeExpandFusionCompute(
    size_t shape_source_input_index, std::vector<ShapeExpandOutputPlan> outputs)
    : shape_source_input_index(shape_source_input_index),
      outputs(std::move(outputs)) {}

OrtStatus* ShapeExpandFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    Ort::ConstValue shape_source = ctx.GetInput(shape_source_input_index);
    std::vector<int64_t> source_shape =
        shape_source.GetTensorTypeAndShapeInfo().GetShape();

    for (const ShapeExpandOutputPlan& plan : outputs) {
      if (plan.kind != ShapeExpandOutputPlan::Kind::Expand) {
        continue;
      }
      Ort::ConstValue input = ctx.GetInput(plan.data_input_index);
      auto input_info = input.GetTensorTypeAndShapeInfo();
      auto elem_type = input_info.GetElementType();
      std::vector<int64_t> input_shape = input_info.GetShape();

      std::vector<int64_t> target_shape;
      for (const ShapeExpandTerm& term : plan.terms) {
        if (term.from_shape_dim) {
          int64_t dim_count = term.dim_count;
          if (dim_count < 0) {
            dim_count =
                static_cast<int64_t>(source_shape.size()) - term.dim_index;
          }
          if (term.dim_index < 0 || dim_count <= 0 ||
              term.dim_index + dim_count >
                  static_cast<int64_t>(source_shape.size())) {
            return Ort::GetApi().CreateStatus(
                ORT_INVALID_ARGUMENT, "ShapeExpand dim index out of range");
          }
          for (int64_t i = 0; i < dim_count; ++i) {
            target_shape.push_back(
                source_shape[static_cast<size_t>(term.dim_index + i)]);
          }
        } else {
          target_shape.insert(target_shape.end(), term.values.begin(),
                              term.values.end());
        }
      }

      if (target_shape.size() < input_shape.size()) {
        target_shape.insert(target_shape.begin(),
                            input_shape.size() - target_shape.size(), 1);
      }
      const size_t offset = target_shape.size() - input_shape.size();
      for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == -1 && i >= offset) {
          target_shape[i] = input_shape[i - offset];
        }
      }
      std::vector<int64_t> out_shape = BroadcastShape(input_shape, target_shape);
      const size_t elem_size = ElementSize(elem_type);
      if (elem_size == 0) {
        return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                          "ShapeExpand unsupported dtype");
      }
      if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
          !CanUseBroadcastKernel(out_shape, input_shape, out_shape)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "ShapeExpand unsupported MUSA input shape");
      }

      Ort::UnownedValue output = ctx.GetOutput(plan.output_index, out_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                          "ShapeExpand requires MUSA output");
      }
      if (plan.has_scalar_fill_value &&
          plan.scalar_fill_value_type == elem_type &&
          NumElements(input_shape) == 1) {
        OrtStatus* status = LaunchStatus(LaunchMusaConstantOfShapeKernel(
            output.GetTensorMutableRawData(), plan.scalar_fill_value_bits,
            static_cast<int32_t>(plan.scalar_fill_value_size),
            NumElements(out_shape), nullptr));
        if (status != nullptr) {
          return status;
        }
        continue;
      }
      OrtStatus* status = LaunchStatus(LaunchMusaExpandKernel(
          input.GetTensorRawData(), output.GetTensorMutableRawData(),
          static_cast<int32_t>(elem_size),
          MakeBroadcastParams(out_shape, input_shape, out_shape), nullptr));
      if (status != nullptr) {
        return status;
      }
      continue;
    }

    for (const ShapeExpandOutputPlan& plan : outputs) {
      if (plan.kind != ShapeExpandOutputPlan::Kind::ConstantOfShape) {
        continue;
      }
      std::vector<int64_t> output_shape;
      for (const ShapeExpandTerm& term : plan.terms) {
        if (term.from_shape_dim) {
          int64_t dim_count = term.dim_count;
          if (dim_count < 0) {
            dim_count = static_cast<int64_t>(source_shape.size()) -
                        term.dim_index;
          }
          if (term.dim_index < 0 || dim_count <= 0 ||
              term.dim_index + dim_count >
                  static_cast<int64_t>(source_shape.size())) {
            return Ort::GetApi().CreateStatus(
                ORT_INVALID_ARGUMENT,
                "ShapeExpand ConstantOfShape dim index out of range");
          }
          for (int64_t i = 0; i < dim_count; ++i) {
            output_shape.push_back(
                source_shape[static_cast<size_t>(term.dim_index + i)]);
          }
        } else {
          output_shape.insert(output_shape.end(), term.values.begin(),
                              term.values.end());
        }
      }
      for (int64_t dim : output_shape) {
        if (dim < 0) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED,
              "ShapeExpand ConstantOfShape requires non-negative dimensions");
        }
      }
      Ort::UnownedValue output = ctx.GetOutput(plan.output_index, output_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "ShapeExpand ConstantOfShape requires MUSA output");
      }
      OrtStatus* status = LaunchStatus(LaunchMusaConstantOfShapeKernel(
          output.GetTensorMutableRawData(), plan.value_bits,
          static_cast<int32_t>(plan.value_size), NumElements(output_shape),
          nullptr));
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

bool IsShapeExpandFusionGraph(Ort::ConstGraph graph) {
  if (!FindShapeNode(graph)) {
    return false;
  }
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Expand") || IsOnnxOp(node, "ConstantOfShape")) {
      return true;
    }
  }
  return false;
}

std::unique_ptr<FusionNodeCompute> CreateShapeExpandFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode shape_node = FindShapeNode(graph);
  if (!shape_node) {
    throw std::runtime_error("ShapeExpand fusion expects a Shape source");
  }
  std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
  if (shape_inputs.size() != 1) {
    throw std::runtime_error("ShapeExpand Shape input mismatch");
  }
  std::vector<ShapeExpandOutputPlan> outputs =
      BuildOutputPlans(graph, fused_node, shape_node);
  if (outputs.empty()) {
    throw std::runtime_error("ShapeExpand fusion has no Expand outputs");
  }
  auto fused_input_indices = FusedInputIndices(fused_node);
  return std::make_unique<ShapeExpandFusionCompute>(
      GetMappedIndex(fused_input_indices, Name(shape_inputs[0]), "input"),
      std::move(outputs));
}
