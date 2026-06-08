// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/shape_reshape_fusion.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/raw_copy_impl.h"
#include "kernels/tensor/tile_impl.h"
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
    throw std::runtime_error("ShapeReshape requires constant shape input");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("ShapeReshape failed to read initializer");
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
  throw std::runtime_error("ShapeReshape constants must be int32/int64");
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
    throw std::runtime_error("unable to map ShapeReshape input " + input_name);
  }
  return it->second;
}

std::optional<size_t> FindFusedInputIndex(
    const std::unordered_map<std::string, size_t>& fused_input_indices,
    const std::string& input_name) {
  auto it = fused_input_indices.find(input_name);
  if (it == fused_input_indices.end()) {
    return std::nullopt;
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
    throw std::runtime_error("unable to map ShapeReshape output " +
                             output_name);
  }
  return it->second;
}

Ort::ConstNode ProducerOf(Ort::ConstValueInfo value_info) {
  Ort::ValueInfoConsumerProducerInfo producer = value_info.GetProducerNode();
  return producer.node;
}

struct ValueConsumer {
  Ort::ConstNode node;
  size_t input_index = 0;
};

std::optional<ValueConsumer> FindSingleConsumerInGraph(
    const std::vector<Ort::ConstNode>& graph_nodes,
    const std::string& value_name) {
  std::optional<ValueConsumer> consumer;
  for (Ort::ConstNode node : graph_nodes) {
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i] || Name(inputs[i]) != value_name) {
        continue;
      }
      if (consumer.has_value()) {
        return std::nullopt;
      }
      consumer = ValueConsumer{node, i};
    }
  }
  return consumer;
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
        throw std::runtime_error("ShapeReshape zero dim exceeds input rank");
      }
      out_shape[i] = input_shape[i];
    }
    if (out_shape[i] == -1) {
      if (infer_idx >= 0) {
        throw std::runtime_error("ShapeReshape only supports one inferred dim");
      }
      infer_idx = static_cast<int64_t>(i);
    } else {
      known *= out_shape[i];
    }
  }
  if (infer_idx >= 0) {
    if (known == 0 || input_size % known != 0) {
      throw std::runtime_error("ShapeReshape cannot infer output dim");
    }
    out_shape[static_cast<size_t>(infer_idx)] = input_size / known;
  }
  return out_shape;
}

MusaTileParams MakeTileParams(const std::vector<int64_t>& input_shape,
                              const std::vector<int64_t>& output_shape) {
  MusaTileParams params{};
  params.rank = static_cast<int32_t>(output_shape.size());
  params.total_elements = NumElements(output_shape);

  std::vector<int64_t> padded_input(output_shape.size(), 1);
  const size_t offset = output_shape.size() - input_shape.size();
  for (size_t i = 0; i < input_shape.size(); ++i) {
    padded_input[offset + i] = input_shape[i];
  }
  std::vector<int64_t> input_strides = Strides(input_shape);

  for (size_t dim = 0; dim < output_shape.size(); ++dim) {
    params.input_dims[dim] = padded_input[dim];
    params.output_dims[dim] = output_shape[dim];
    if (dim < offset) {
      params.input_strides[dim] = 0;
    } else {
      params.input_strides[dim] = input_strides[dim - offset];
    }
  }
  return params;
}

std::vector<int64_t> ResolveTileOutputShape(
    const std::vector<int64_t>& input_shape, std::vector<int64_t>& repeats) {
  if (repeats.size() < input_shape.size()) {
    repeats.insert(repeats.begin(), input_shape.size() - repeats.size(), 1);
  }
  if (repeats.size() > kMusaMaxBroadcastRank) {
    throw std::runtime_error("ShapeReshape Tile rank exceeds MUSA kernel limit");
  }

  std::vector<int64_t> padded_input(repeats.size(), 1);
  const size_t offset = repeats.size() - input_shape.size();
  for (size_t i = 0; i < input_shape.size(); ++i) {
    padded_input[offset + i] = input_shape[i];
  }

  std::vector<int64_t> output_shape(repeats.size(), 1);
  for (size_t i = 0; i < repeats.size(); ++i) {
    if (repeats[i] < 0) {
      throw std::runtime_error("ShapeReshape Tile requires non-negative repeats");
    }
    output_shape[i] = padded_input[i] * repeats[i];
  }
  return output_shape;
}

Ort::ConstNode FindShapeNode(Ort::ConstGraph graph) {
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Shape")) {
      return node;
    }
  }
  return Ort::ConstNode{nullptr};
}

std::vector<int64_t> SplitSizes(Ort::ConstNode split_node,
                                size_t split_output_count,
                                size_t shape_rank) {
  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  if (split_inputs.size() > 1) {
    return ReadIntInitializer(split_inputs[1]);
  }
  if (split_output_count == 0 || shape_rank % split_output_count != 0) {
    throw std::runtime_error("ShapeReshape unsupported Split num_outputs");
  }
  return std::vector<int64_t>(split_output_count,
                              static_cast<int64_t>(shape_rank /
                                                   split_output_count));
}

ShapeReshapeTerm TermFromSplitOutput(Ort::ConstNode split_node,
                                     const std::string& output_name,
                                     size_t shape_rank) {
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  std::vector<int64_t> split_sizes =
      SplitSizes(split_node, split_outputs.size(), shape_rank);
  if (split_sizes.size() != split_outputs.size()) {
    throw std::runtime_error("ShapeReshape Split size mismatch");
  }
  int64_t split_offset = 0;
  for (size_t i = 0; i < split_outputs.size(); ++i) {
    if (Name(split_outputs[i]) == output_name) {
      if (split_sizes[i] != 1) {
        throw std::runtime_error(
            "ShapeReshape only supports scalar Split dims");
      }
      return ShapeReshapeTerm{true, split_offset, 1, {}};
    }
    split_offset += split_sizes[i];
  }
  throw std::runtime_error("ShapeReshape Split output not found");
}

ShapeReshapeTerm TermFromSliceOutput(Ort::ConstNode slice_node,
                                     const std::string& output_name,
                                     std::optional<size_t> shape_rank) {
  std::vector<Ort::ConstValueInfo> slice_outputs = slice_node.GetOutputs();
  if (slice_outputs.size() != 1 || Name(slice_outputs[0]) != output_name) {
    throw std::runtime_error("ShapeReshape Slice output mismatch");
  }

  std::vector<Ort::ConstValueInfo> slice_inputs = slice_node.GetInputs();
  if (slice_inputs.size() < 3 || slice_inputs.size() > 5) {
    throw std::runtime_error("ShapeReshape unsupported Slice input count");
  }
  std::vector<int64_t> starts = ReadIntInitializer(slice_inputs[1]);
  std::vector<int64_t> ends = ReadIntInitializer(slice_inputs[2]);
  if (starts.size() != 1 || ends.size() != 1) {
    throw std::runtime_error("ShapeReshape Slice must select one range");
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
    throw std::runtime_error("ShapeReshape Slice must use axis 0 step 1");
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
        "ShapeReshape dynamic-rank Slice requires non-negative bounds");
  }
  if (end <= start) {
    throw std::runtime_error("ShapeReshape Slice selects empty shape range");
  }
  return ShapeReshapeTerm{true, start, end - start, {}};
}

std::vector<ShapeReshapeTerm> BuildTerms(Ort::ConstNode concat_node,
                                         Ort::ConstNode shape_node) {
  std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
  if (shape_inputs.size() != 1) {
    throw std::runtime_error("ShapeReshape Shape input mismatch");
  }
  auto shape_source_shape = GetTensorShape(shape_inputs[0]);

  std::vector<ShapeReshapeTerm> terms;
  for (Ort::ConstValueInfo input : concat_node.GetInputs()) {
    if (input.IsConstantInitializer()) {
      ShapeReshapeTerm term;
      term.values = ReadIntInitializer(input);
      terms.push_back(std::move(term));
      continue;
    }

    Ort::ConstNode producer = ProducerOf(input);
    if (producer && IsOnnxOp(producer, "Split")) {
      if (!shape_source_shape.has_value()) {
        throw std::runtime_error("ShapeReshape Split requires static shape rank");
      }
      terms.push_back(TermFromSplitOutput(producer, Name(input),
                                          shape_source_shape->size()));
      continue;
    }
    if (producer && IsOnnxOp(producer, "Slice")) {
      terms.push_back(TermFromSliceOutput(producer, Name(input),
                                          shape_source_shape.has_value()
                                              ? std::optional<size_t>(
                                                    shape_source_shape->size())
                                              : std::nullopt));
      continue;
    }

    ShapeReshapeTerm term;
    term.values = ReadIntInitializer(input);
    terms.push_back(std::move(term));
  }
  return terms;
}

bool FindFinalCastConcat(Ort::ConstValueInfo input,
                         Ort::ConstNode& concat_node) {
  Ort::ConstNode final_cast_node = ProducerOf(input);
  if (!final_cast_node || !IsOnnxOp(final_cast_node, "Cast") ||
      ReadIntAttribute(final_cast_node, "to", 0) !=
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> final_cast_inputs =
      final_cast_node.GetInputs();
  if (final_cast_inputs.size() != 1) {
    return false;
  }
  Ort::ConstNode candidate = ProducerOf(final_cast_inputs[0]);
  if (!candidate || !IsOnnxOp(candidate, "Concat") ||
      ReadIntAttribute(candidate, "axis", 0) != 0) {
    return false;
  }
  concat_node = candidate;
  return true;
}

std::vector<ShapeReshapeOutputPlan> BuildOutputPlans(
    Ort::ConstGraph graph, Ort::ConstNode fused_node, Ort::ConstNode shape_node) {
  auto fused_input_indices = FusedInputIndices(fused_node);
  auto fused_output_indices = FusedOutputIndices(fused_node);
  std::vector<Ort::ConstNode> graph_nodes = graph.GetNodes();
  std::unordered_set<size_t> tile_reshape_ids;
  std::vector<ShapeReshapeOutputPlan> outputs;

  for (Ort::ConstNode node : graph_nodes) {
    if (!IsOnnxOp(node, "Tile")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> node_outputs = node.GetOutputs();
    if (inputs.size() != 2 || node_outputs.size() != 1) {
      continue;
    }

    Ort::ConstNode repeats_concat_node{nullptr};
    if (!FindFinalCastConcat(inputs[1], repeats_concat_node)) {
      continue;
    }

    ShapeReshapeOutputPlan plan;
    plan.kind = ShapeReshapeOutputPlan::Kind::kTile;
    plan.data_input_index =
        GetFusedInputIndex(fused_input_indices, Name(inputs[0]));
    plan.terms = BuildTerms(repeats_concat_node, shape_node);

    std::optional<ValueConsumer> consumer =
        FindSingleConsumerInGraph(graph_nodes, Name(node_outputs[0]));
    if (consumer.has_value() && consumer->input_index == 0 &&
        IsOnnxOp(consumer->node, "Reshape")) {
      Ort::ConstNode reshape_node = consumer->node;
      std::vector<Ort::ConstValueInfo> reshape_inputs =
          reshape_node.GetInputs();
      std::vector<Ort::ConstValueInfo> reshape_outputs =
          reshape_node.GetOutputs();
      Ort::ConstNode reshape_concat_node{nullptr};
      if (reshape_inputs.size() == 2 && reshape_outputs.size() == 1 &&
          FindFinalCastConcat(reshape_inputs[1], reshape_concat_node)) {
        plan.kind = ShapeReshapeOutputPlan::Kind::kTileReshape;
        plan.output_index =
            GetFusedOutputIndex(fused_output_indices, Name(reshape_outputs[0]));
        plan.tile_terms = std::move(plan.terms);
        plan.terms = BuildTerms(reshape_concat_node, shape_node);
        plan.allowzero = ReadIntAttribute(reshape_node, "allowzero", 0);
        tile_reshape_ids.insert(reshape_node.GetId());
      }
    }
    if (plan.kind == ShapeReshapeOutputPlan::Kind::kTile) {
      plan.output_index =
          GetFusedOutputIndex(fused_output_indices, Name(node_outputs[0]));
    }
    outputs.push_back(std::move(plan));
  }

  for (Ort::ConstNode node : graph_nodes) {
    const bool is_reshape = IsOnnxOp(node, "Reshape");
    if (!is_reshape || tile_reshape_ids.count(node.GetId()) != 0) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> node_outputs = node.GetOutputs();
    if (inputs.size() != 2 || node_outputs.size() != 1) {
      continue;
    }

    Ort::ConstNode concat_node{nullptr};
    if (!FindFinalCastConcat(inputs[1], concat_node)) {
      continue;
    }

    std::optional<size_t> data_input_index =
        FindFusedInputIndex(fused_input_indices, Name(inputs[0]));
    if (!data_input_index.has_value()) {
      Ort::ConstNode data_producer = ProducerOf(inputs[0]);
      if (data_producer && IsOnnxOp(data_producer, "Tile")) {
        continue;
      }
      continue;
    }
    ShapeReshapeOutputPlan plan;
    plan.kind = ShapeReshapeOutputPlan::Kind::kReshape;
    plan.data_input_index = *data_input_index;
    plan.output_index =
        GetFusedOutputIndex(fused_output_indices, Name(node_outputs[0]));
    plan.terms = BuildTerms(concat_node, shape_node);
    plan.allowzero = ReadIntAttribute(node, "allowzero", 0);
    outputs.push_back(std::move(plan));
  }
  return outputs;
}

}  // namespace

ShapeReshapeFusionCompute::ShapeReshapeFusionCompute(
    size_t shape_source_input_index, std::vector<ShapeReshapeOutputPlan> outputs)
    : shape_source_input_index(shape_source_input_index),
      outputs(std::move(outputs)) {}

std::vector<int64_t> ResolveShapeTerms(
    const std::vector<ShapeReshapeTerm>& terms,
    const std::vector<int64_t>& source_shape) {
  std::vector<int64_t> values;
  for (const ShapeReshapeTerm& term : terms) {
    if (term.from_shape_dim) {
      int64_t dim_count = term.dim_count;
      if (dim_count < 0) {
        dim_count = static_cast<int64_t>(source_shape.size()) - term.dim_index;
      }
      if (term.dim_index < 0 || dim_count <= 0 ||
          term.dim_index + dim_count >
              static_cast<int64_t>(source_shape.size())) {
        throw std::runtime_error("ShapeReshape dim index out of range");
      }
      for (int64_t i = 0; i < dim_count; ++i) {
        values.push_back(source_shape[static_cast<size_t>(term.dim_index + i)]);
      }
    } else {
      values.insert(values.end(), term.values.begin(), term.values.end());
    }
  }
  return values;
}

OrtStatus* ShapeReshapeFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    Ort::ConstValue shape_source = ctx.GetInput(shape_source_input_index);
    std::vector<int64_t> source_shape =
        shape_source.GetTensorTypeAndShapeInfo().GetShape();

    if (outputs.size() == 3 &&
        std::all_of(outputs.begin(), outputs.end(),
                    [](const ShapeReshapeOutputPlan& plan) {
                      return plan.kind == ShapeReshapeOutputPlan::Kind::kReshape;
                    })) {
      std::array<const float*, 3> input_data{};
      std::array<float*, 3> output_data{};
      int64_t element_count = -1;
      bool can_use_copy3 = true;

      for (size_t i = 0; i < outputs.size(); ++i) {
        const ShapeReshapeOutputPlan& plan = outputs[i];
        Ort::ConstValue data = ctx.GetInput(plan.data_input_index);
        auto data_info = data.GetTensorTypeAndShapeInfo();
        if (data_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            !IsGpuMemory(data.GetTensorMemoryInfo())) {
          can_use_copy3 = false;
          break;
        }

        std::vector<int64_t> data_shape = data_info.GetShape();
        std::vector<int64_t> requested_shape =
            ResolveShapeTerms(plan.terms, source_shape);
        std::vector<int64_t> output_shape = ResolveReshapeOutputShape(
            data_shape, requested_shape, plan.allowzero);
        const int64_t current_element_count = NumElements(data_shape);
        if (NumElements(output_shape) != current_element_count ||
            (element_count >= 0 &&
             element_count != current_element_count)) {
          can_use_copy3 = false;
          break;
        }

        Ort::UnownedValue output =
            ctx.GetOutput(plan.output_index, output_shape);
        if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
          can_use_copy3 = false;
          break;
        }
        input_data[i] = data.GetTensorData<float>();
        output_data[i] = output.GetTensorMutableData<float>();
        element_count = current_element_count;
      }

      if (can_use_copy3 && element_count >= 0) {
        return LaunchStatus(LaunchMusaRawCopy3Float(
            input_data[0], input_data[1], input_data[2], output_data[0],
            output_data[1], output_data[2], element_count, nullptr));
      }
    }

    for (const ShapeReshapeOutputPlan& plan : outputs) {
      Ort::ConstValue data = ctx.GetInput(plan.data_input_index);
      std::vector<int64_t> data_shape =
          data.GetTensorTypeAndShapeInfo().GetShape();

      if (plan.kind == ShapeReshapeOutputPlan::Kind::kTile ||
          plan.kind == ShapeReshapeOutputPlan::Kind::kTileReshape) {
        std::vector<int64_t> repeats = ResolveShapeTerms(
            plan.kind == ShapeReshapeOutputPlan::Kind::kTile
                ? plan.terms
                : plan.tile_terms,
            source_shape);
        std::vector<int64_t> tile_output_shape =
            ResolveTileOutputShape(data_shape, repeats);
        std::vector<int64_t> output_shape = tile_output_shape;
        if (plan.kind == ShapeReshapeOutputPlan::Kind::kTileReshape) {
          std::vector<int64_t> requested_shape =
              ResolveShapeTerms(plan.terms, source_shape);
          output_shape = ResolveReshapeOutputShape(
              tile_output_shape, requested_shape, plan.allowzero);
          if (NumElements(output_shape) != NumElements(tile_output_shape)) {
            return Ort::GetApi().CreateStatus(
                ORT_EP_FAIL,
                "ShapeReshape TileReshape element count mismatch");
          }
        }

        auto elem_type = data.GetTensorTypeAndShapeInfo().GetElementType();
        const size_t elem_size = ElementSize(elem_type);
        if (elem_size == 0) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "ShapeReshape Tile unsupported dtype");
        }
        if (!IsGpuMemory(data.GetTensorMemoryInfo())) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "ShapeReshape Tile requires MUSA input");
        }
        Ort::UnownedValue output =
            ctx.GetOutput(plan.output_index, output_shape);
        if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "ShapeReshape Tile requires MUSA output");
        }
        OrtStatus* status = nullptr;
        if (tile_output_shape == data_shape) {
          status = CopyRawTensor(data, output, data.GetTensorSizeInBytes());
        } else if (!data_shape.empty() && data_shape.back() > 0 &&
                   repeats.back() > 1 &&
                   std::all_of(repeats.begin(), repeats.end() - 1,
                               [](int64_t repeat) { return repeat == 1; })) {
          const int64_t cols = data_shape.back();
          const int64_t rows = NumElements(data_shape) / cols;
          status = LaunchStatus(LaunchMusaTileLastDimKernel(
              data.GetTensorRawData(), output.GetTensorMutableRawData(),
              static_cast<int32_t>(elem_size), rows, cols, repeats.back(),
              nullptr));
        } else {
          status = LaunchStatus(LaunchMusaTileKernel(
              data.GetTensorRawData(), output.GetTensorMutableRawData(),
              static_cast<int32_t>(elem_size),
              MakeTileParams(data_shape, tile_output_shape), nullptr));
        }
        if (status != nullptr) {
          return status;
        }
        continue;
      }

      std::vector<int64_t> requested_shape =
          ResolveShapeTerms(plan.terms, source_shape);
      std::vector<int64_t> output_shape =
          ResolveReshapeOutputShape(data_shape, requested_shape, plan.allowzero);
      Ort::UnownedValue output = ctx.GetOutput(plan.output_index, output_shape);
      OrtStatus* status =
          CopyRawTensor(data, output, data.GetTensorSizeInBytes());
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

bool IsShapeReshapeFusionGraph(Ort::ConstGraph graph) {
  Ort::ConstNode shape_node = FindShapeNode(graph);
  if (!shape_node) {
    return false;
  }
  try {
    size_t reshape_count = 0;
    for (Ort::ConstNode node : graph.GetNodes()) {
      if (IsOnnxOp(node, "Reshape")) {
        ++reshape_count;
      }
    }
    return reshape_count > 0;
  } catch (const std::exception&) {
    return false;
  }
}

std::unique_ptr<FusionNodeCompute> CreateShapeReshapeFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode shape_node = FindShapeNode(graph);
  if (!shape_node) {
    throw std::runtime_error(
        "ShapeReshape fusion expects a Shape metadata source");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
  if (shape_inputs.size() != 1) {
    throw std::runtime_error("ShapeReshape Shape input mismatch");
  }
  std::vector<ShapeReshapeOutputPlan> outputs =
      BuildOutputPlans(graph, fused_node, shape_node);
  if (outputs.empty()) {
    throw std::runtime_error("ShapeReshape fusion has no Reshape outputs");
  }
  return std::make_unique<ShapeReshapeFusionCompute>(
      GetFusedInputIndex(fused_input_indices, Name(shape_inputs[0])),
      std::move(outputs));
}
