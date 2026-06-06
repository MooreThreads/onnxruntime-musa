// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/shape_reshape_fusion.h"

#include <climits>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/shared_inc/op_kernel_common.h"

/*
 * ShapeReshape Fusion Pattern
 *
 * Conservatively fuses the shape-metadata suffix:
 *
 *   Shape(S) -> [Gather(initializer indices)] -> Cast(int32/int64)
 *     -> Gather(initializer indices)
 *     -> Concat(gathered dims, small initializer suffixes)
 *     -> Cast(int64)
 *     -> one or more Reshape(Data_i, shape)
 *
 * The fused compute does not materialize intermediate shape tensors. It reads
 * only host-side tensor metadata from S, combines it with compile-time copied
 * initializer indices/suffixes, allocates each Reshape output, and keeps the
 * data tensor copy/alias on the normal MUSA path.
 */

namespace {

enum class ShapePartKind { GatheredShape, ConstantValues };

struct ShapePart {
  ShapePartKind kind;
  std::vector<int64_t> values;
};

struct TargetShapeSpec {
  std::vector<ShapePart> parts;
};

struct ReshapePlan {
  size_t data_input_index;
  size_t output_index;
  int64_t allowzero;
  TargetShapeSpec target_shape;
};

struct ShapeReshapeFusionCompute : FusionNodeCompute {
  ShapeReshapeFusionCompute(size_t shape_input_index,
                            bool shape_input_is_shape_tensor,
                            std::vector<int64_t> pre_gather_indices,
                            std::vector<int64_t> gather_indices,
                            int64_t shape_cast_to,
                            std::vector<ReshapePlan> reshapes)
      : shape_input_index(shape_input_index),
        shape_input_is_shape_tensor(shape_input_is_shape_tensor),
        pre_gather_indices(std::move(pre_gather_indices)),
        gather_indices(std::move(gather_indices)),
        shape_cast_to(shape_cast_to),
        reshapes(std::move(reshapes)) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue shape_source = ctx.GetInput(shape_input_index);
      std::vector<int64_t> source_shape =
          shape_input_is_shape_tensor
              ? ReadShapeTensor(shape_source)
              : shape_source.GetTensorTypeAndShapeInfo().GetShape();
      if (!pre_gather_indices.empty()) {
        source_shape = GatherValues(source_shape, pre_gather_indices);
      }
      std::vector<int64_t> gathered_shape = GatherShape(source_shape);

      for (const ReshapePlan& reshape : reshapes) {
        Ort::ConstValue data = ctx.GetInput(reshape.data_input_index);
        auto data_info = data.GetTensorTypeAndShapeInfo();
        if (data_info.GetElementType() ==
            ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED,
              "ShapeReshape fusion does not support string data tensors");
        }

        std::vector<int64_t> target =
            BuildTargetShape(reshape.target_shape, gathered_shape);
        std::vector<int64_t> output_shape = ResolveReshapeOutputShape(
            data_info.GetShape(), std::move(target), reshape.allowzero);
        Ort::UnownedValue output =
            ctx.GetOutput(reshape.output_index, output_shape);
        RETURN_IF_ERROR(
            CopyRawTensor(data, output, data.GetTensorSizeInBytes()));
      }

      return nullptr;
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  std::vector<int64_t> GatherShape(
      const std::vector<int64_t>& source_shape) const {
    std::vector<int64_t> result = GatherValues(source_shape, gather_indices);
    if (shape_cast_to == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      for (int64_t& value : result) {
        value = static_cast<int64_t>(static_cast<int32_t>(value));
      }
    } else if (shape_cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      throw std::runtime_error("ShapeReshape unsupported Shape Cast target");
    }
    return result;
  }

  static std::vector<int64_t> GatherValues(
      const std::vector<int64_t>& source_shape,
      const std::vector<int64_t>& indices) {
    std::vector<int64_t> result;
    result.reserve(indices.size());
    const int64_t rank = static_cast<int64_t>(source_shape.size());
    for (int64_t raw_index : indices) {
      int64_t index = raw_index < 0 ? raw_index + rank : raw_index;
      if (index < 0 || index >= rank) {
        throw std::runtime_error("ShapeReshape Gather index is out of range");
      }

      result.push_back(source_shape[static_cast<size_t>(index)]);
    }
    return result;
  }

  static std::vector<int64_t> ReadShapeTensor(Ort::ConstValue value) {
    auto info = value.GetTensorTypeAndShapeInfo();
    const size_t count = static_cast<size_t>(info.GetElementCount());
    if (count > 100) {
      throw std::runtime_error("ShapeReshape shape tensor input is too large");
    }

    std::vector<uint8_t> bytes;
    Ort::Status copy_status{CopyToHost(value, bytes)};
    if (!copy_status.IsOK()) {
      throw Ort::Exception(copy_status.GetErrorMessage(),
                           copy_status.GetErrorCode());
    }
    std::vector<int64_t> result;
    result.reserve(count);
    ONNXTensorElementDataType elem_type = info.GetElementType();
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      const int32_t* data = reinterpret_cast<const int32_t*>(bytes.data());
      for (size_t i = 0; i < count; ++i) {
        result.push_back(static_cast<int64_t>(data[i]));
      }
    } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const int64_t* data = reinterpret_cast<const int64_t*>(bytes.data());
      result.assign(data, data + count);
    } else {
      throw std::runtime_error(
          "ShapeReshape shape tensor input must be int32/int64");
    }
    return result;
  }

  static std::vector<int64_t> BuildTargetShape(
      const TargetShapeSpec& spec, const std::vector<int64_t>& gathered_shape) {
    std::vector<int64_t> target_shape;
    for (const ShapePart& part : spec.parts) {
      const std::vector<int64_t>& values =
          part.kind == ShapePartKind::GatheredShape ? gathered_shape
                                                    : part.values;
      target_shape.insert(target_shape.end(), values.begin(), values.end());
    }
    return target_shape;
  }

  static std::vector<int64_t> ResolveReshapeOutputShape(
      const std::vector<int64_t>& input_shape, std::vector<int64_t> out_shape,
      int64_t allowzero) {
    int64_t input_size = NumElements(input_shape);
    int64_t known = 1;
    int infer_idx = -1;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      if (out_shape[i] == 0 && !allowzero) {
        out_shape[i] = input_shape[i];
      }
      if (out_shape[i] == -1) {
        infer_idx = static_cast<int>(i);
      } else {
        known *= out_shape[i];
      }
    }
    if (infer_idx >= 0) {
      out_shape[static_cast<size_t>(infer_idx)] = input_size / known;
    }
    return out_shape;
  }

  size_t shape_input_index;
  bool shape_input_is_shape_tensor;
  std::vector<int64_t> pre_gather_indices;
  std::vector<int64_t> gather_indices;
  int64_t shape_cast_to;
  std::vector<ReshapePlan> reshapes;
};

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(const Ort::ConstNode& node, const char* op_type) {
  return node.GetOperatorType() == op_type && IsOnnxDomain(node.GetDomain());
}

std::string Name(Ort::ConstValueInfo value_info) {
  if (value_info == nullptr) {
    return {};
  }
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
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    throw std::runtime_error("ShapeReshape expected a constant initializer");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK()) {
    throw std::runtime_error("ShapeReshape failed to read initializer " +
                             Name(value_info));
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  const size_t count = static_cast<size_t>(info.GetElementCount());
  if (count > 100) {
    throw std::runtime_error("ShapeReshape initializer is too large");
  }

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
        "ShapeReshape only supports int32/int64 initializers");
  }
  return result;
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

size_t GetIndex(const std::unordered_map<std::string, size_t>& indices,
                const std::string& name, const char* kind) {
  auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error(std::string("unable to map ShapeReshape ") + kind +
                             " " + name);
  }
  return it->second;
}

using ConsumerMap =
    std::unordered_map<std::string,
                       std::vector<std::pair<Ort::ConstNode, size_t>>>;

ConsumerMap BuildConsumers(const std::vector<Ort::ConstNode>& nodes) {
  ConsumerMap consumers;
  for (Ort::ConstNode node : nodes) {
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      const std::string input_name = Name(inputs[i]);
      if (!input_name.empty()) {
        consumers[input_name].push_back({node, i});
      }
    }
  }
  return consumers;
}

std::unordered_map<std::string, Ort::ConstNode> BuildProducers(
    const std::vector<Ort::ConstNode>& nodes) {
  std::unordered_map<std::string, Ort::ConstNode> producers;
  for (Ort::ConstNode node : nodes) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      const std::string output_name = Name(output);
      if (!output_name.empty()) {
        producers.emplace(output_name, node);
      }
    }
  }
  return producers;
}

Ort::ConstNode ProducerFor(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::string& name) {
  auto it = producers.find(name);
  if (it == producers.end()) {
    return Ort::ConstNode{nullptr};
  }
  return it->second;
}

TargetShapeSpec BuildTargetShapeSpec(Ort::ConstNode concat_node,
                                     const std::string& gather_output_name) {
  TargetShapeSpec spec;
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  for (Ort::ConstValueInfo input : concat_inputs) {
    if (Name(input) == gather_output_name) {
      spec.parts.push_back({ShapePartKind::GatheredShape, {}});
    } else {
      spec.parts.push_back(
          {ShapePartKind::ConstantValues, ReadIntInitializer(input)});
    }
  }
  return spec;
}

}  // namespace

bool IsShapeReshapeFusionGraph(Ort::ConstGraph graph) {
  bool has_shape = false;
  bool has_gather = false;
  bool has_concat = false;
  bool has_reshape = false;
  for (Ort::ConstNode node : graph.GetNodes()) {
    has_shape = has_shape || IsOnnxOp(node, "Shape");
    has_gather = has_gather || IsOnnxOp(node, "Gather");
    has_concat = has_concat || IsOnnxOp(node, "Concat");
    has_reshape = has_reshape || IsOnnxOp(node, "Reshape");
  }
  return has_gather && has_concat && has_reshape;
}

std::unique_ptr<FusionNodeCompute> CreateShapeReshapeFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  auto producers = BuildProducers(nodes);
  auto consumers = BuildConsumers(nodes);
  auto fused_input_indices = FusedInputIndices(fused_node);
  auto fused_output_indices = FusedOutputIndices(fused_node);

  Ort::ConstNode gather_node{nullptr};
  for (Ort::ConstNode node : nodes) {
    if (!IsOnnxOp(node, "Gather")) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
    if (outputs.size() != 1) {
      continue;
    }

    bool feeds_concat = false;
    for (const auto& consumer : consumers[Name(outputs[0])]) {
      feeds_concat = feeds_concat || IsOnnxOp(consumer.first, "Concat");
    }
    if (!feeds_concat) {
      continue;
    }

    if (gather_node) {
      throw std::runtime_error("ShapeReshape expects one shape Gather node");
    }
    gather_node = node;
  }
  if (!gather_node) {
    throw std::runtime_error("ShapeReshape missing shape Gather node");
  }

  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gather_outputs = gather_node.GetOutputs();
  if (gather_inputs.size() != 2 || gather_outputs.size() != 1 ||
      ReadIntAttribute(gather_node, "axis", 0) != 0) {
    throw std::runtime_error("invalid ShapeReshape Gather node");
  }

  Ort::ConstNode shape_cast_node =
      ProducerFor(producers, Name(gather_inputs[0]));
  if (!shape_cast_node || !IsOnnxOp(shape_cast_node, "Cast")) {
    throw std::runtime_error("ShapeReshape missing Shape Cast node");
  }
  const int64_t shape_cast_to = ReadIntAttribute(shape_cast_node, "to", -1);
  if (shape_cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
      shape_cast_to != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::runtime_error("ShapeReshape unsupported Shape Cast target");
  }

  std::vector<Ort::ConstValueInfo> shape_cast_inputs =
      shape_cast_node.GetInputs();
  if (shape_cast_inputs.size() != 1) {
    throw std::runtime_error("invalid ShapeReshape Shape Cast node");
  }
  std::vector<int64_t> pre_gather_indices;
  Ort::ConstNode shape_input_producer =
      ProducerFor(producers, Name(shape_cast_inputs[0]));
  bool shape_input_is_shape_tensor = true;
  std::string shape_input_name = Name(shape_cast_inputs[0]);
  if (shape_input_producer) {
    if (IsOnnxOp(shape_input_producer, "Shape")) {
      std::vector<Ort::ConstValueInfo> shape_inputs =
          shape_input_producer.GetInputs();
      if (shape_inputs.size() != 1) {
        throw std::runtime_error("invalid ShapeReshape Shape node");
      }
      shape_input_is_shape_tensor = false;
      shape_input_name = Name(shape_inputs[0]);
    } else if (IsOnnxOp(shape_input_producer, "Gather")) {
      std::vector<Ort::ConstValueInfo> pre_gather_inputs =
          shape_input_producer.GetInputs();
      std::vector<Ort::ConstValueInfo> pre_gather_outputs =
          shape_input_producer.GetOutputs();
      if (pre_gather_inputs.size() != 2 || pre_gather_outputs.size() != 1 ||
          ReadIntAttribute(shape_input_producer, "axis", 0) != 0) {
        throw std::runtime_error("invalid ShapeReshape pre-Gather node");
      }
      pre_gather_indices = ReadIntInitializer(pre_gather_inputs[1]);
      shape_input_name = Name(pre_gather_inputs[0]);

      Ort::ConstNode shape_node =
          ProducerFor(producers, Name(pre_gather_inputs[0]));
      if (shape_node) {
        if (!IsOnnxOp(shape_node, "Shape")) {
          throw std::runtime_error(
              "ShapeReshape pre-Gather input must be Shape output");
        }
        std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
        if (shape_inputs.size() != 1) {
          throw std::runtime_error("invalid ShapeReshape Shape node");
        }
        shape_input_is_shape_tensor = false;
        shape_input_name = Name(shape_inputs[0]);
      }
    } else {
      throw std::runtime_error(
          "ShapeReshape Cast input must be Shape or pre-Gather output");
    }
  }

  const size_t shape_input_index =
      GetIndex(fused_input_indices, shape_input_name, "input");
  std::vector<int64_t> gather_indices = ReadIntInitializer(gather_inputs[1]);

  const std::string gather_output_name = Name(gather_outputs[0]);
  std::vector<ReshapePlan> reshapes;
  for (const auto& concat_consumer : consumers[gather_output_name]) {
    Ort::ConstNode concat_node = concat_consumer.first;
    if (!IsOnnxOp(concat_node, "Concat") ||
        ReadIntAttribute(concat_node, "axis", 0) != 0) {
      throw std::runtime_error("ShapeReshape Gather must feed Concat(axis=0)");
    }

    TargetShapeSpec target_shape =
        BuildTargetShapeSpec(concat_node, gather_output_name);
    std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
    if (concat_outputs.size() != 1) {
      throw std::runtime_error("invalid ShapeReshape Concat node");
    }

    const std::string concat_output_name = Name(concat_outputs[0]);
    auto concat_output_consumers = consumers[concat_output_name];
    if (concat_output_consumers.size() != 1 ||
        concat_output_consumers[0].second != 0 ||
        !IsOnnxOp(concat_output_consumers[0].first, "Cast")) {
      throw std::runtime_error("ShapeReshape Concat must feed final Cast");
    }

    Ort::ConstNode final_cast_node = concat_output_consumers[0].first;
    if (ReadIntAttribute(final_cast_node, "to", -1) !=
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      throw std::runtime_error("ShapeReshape final Cast must target int64");
    }

    std::vector<Ort::ConstValueInfo> final_cast_outputs =
        final_cast_node.GetOutputs();
    if (final_cast_outputs.size() != 1) {
      throw std::runtime_error("invalid ShapeReshape final Cast node");
    }

    const std::string final_shape_name = Name(final_cast_outputs[0]);
    for (const auto& reshape_consumer : consumers[final_shape_name]) {
      Ort::ConstNode reshape_node = reshape_consumer.first;
      if (reshape_consumer.second != 1 || !IsOnnxOp(reshape_node, "Reshape")) {
        throw std::runtime_error("ShapeReshape final Cast must feed Reshape");
      }

      std::vector<Ort::ConstValueInfo> reshape_inputs =
          reshape_node.GetInputs();
      std::vector<Ort::ConstValueInfo> reshape_outputs =
          reshape_node.GetOutputs();
      if (reshape_inputs.size() != 2 || reshape_outputs.size() != 1) {
        throw std::runtime_error("invalid ShapeReshape Reshape node");
      }

      reshapes.push_back(
          {GetIndex(fused_input_indices, Name(reshape_inputs[0]), "input"),
           GetIndex(fused_output_indices, Name(reshape_outputs[0]), "output"),
           ReadIntAttribute(reshape_node, "allowzero", 0), target_shape});
    }
  }

  if (reshapes.empty()) {
    throw std::runtime_error("ShapeReshape fusion has no Reshape consumers");
  }

  return std::make_unique<ShapeReshapeFusionCompute>(
      shape_input_index, shape_input_is_shape_tensor,
      std::move(pre_gather_indices), std::move(gather_indices), shape_cast_to,
      std::move(reshapes));
}
