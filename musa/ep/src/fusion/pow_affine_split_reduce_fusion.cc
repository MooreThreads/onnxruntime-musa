// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/pow_affine_split_reduce_fusion.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/reduction/pow_affine_split_reduce_impl.h"
#include "kernels/shared_inc/op_kernel_common.h"

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
    throw std::runtime_error(
        "PowAffineSplitReduce requires constant int initializer");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error(
        "PowAffineSplitReduce failed to read initializer");
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  const auto elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return ReadTyped<int64_t>(value);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> values = ReadTyped<int32_t>(value);
    return std::vector<int64_t>(values.begin(), values.end());
  }
  throw std::runtime_error("PowAffineSplitReduce requires int32/int64 values");
}

bool AxesInputIsAxis1(Ort::ConstNode node) {
  std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  if (inputs.size() < 2 || !inputs[1].IsConstantInitializer()) {
    return false;
  }
  std::vector<int64_t> axes = ReadIntInitializer(inputs[1]);
  return axes.size() == 1 && axes[0] == 1;
}

MusaSplitReduceMode ReduceModeForNode(Ort::ConstNode node) {
  if (IsOnnxOp(node, "ReduceProd")) {
    return MusaSplitReduceMode::Prod;
  }
  if (IsOnnxOp(node, "ReduceMean")) {
    return MusaSplitReduceMode::Mean;
  }
  throw std::runtime_error("PowAffineSplitReduce unsupported reduce op");
}

Ort::ConstNode ProducerNode(Ort::ConstValueInfo value_info) {
  return value_info.GetProducerNode().node;
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
    throw std::runtime_error(std::string("unable to map PowAffineSplitReduce ") +
                             kind + " " + name);
  }
  return it->second;
}

Ort::ConstNode FindSplitNode(Ort::ConstGraph graph) {
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Split")) {
      return node;
    }
  }
  return Ort::ConstNode{nullptr};
}

std::array<int64_t, 3> BroadcastStridesForRank3(
    const std::vector<int64_t>& operand_shape,
    const std::vector<int64_t>& output_shape) {
  if (output_shape.size() != 3 || operand_shape.size() > 3) {
    throw std::runtime_error("PowAffineSplitReduce requires rank <= 3 inputs");
  }

  std::array<int64_t, 3> strides = {0, 0, 0};
  int64_t contiguous_stride = 1;
  std::vector<int64_t> operand_strides(operand_shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(operand_shape.size()) - 1; i >= 0;
       --i) {
    operand_strides[static_cast<size_t>(i)] = contiguous_stride;
    contiguous_stride *= operand_shape[static_cast<size_t>(i)];
  }

  const int64_t rank_delta =
      static_cast<int64_t>(output_shape.size() - operand_shape.size());
  for (size_t dim = 0; dim < output_shape.size(); ++dim) {
    const int64_t operand_dim_index = static_cast<int64_t>(dim) - rank_delta;
    if (operand_dim_index < 0) {
      strides[dim] = 0;
      continue;
    }

    const int64_t operand_dim =
        operand_shape[static_cast<size_t>(operand_dim_index)];
    const int64_t output_dim = output_shape[dim];
    if (operand_dim == 1) {
      strides[dim] = 0;
      continue;
    }
    if (operand_dim != output_dim) {
      throw std::runtime_error(
          "PowAffineSplitReduce input is not broadcastable");
    }
    strides[dim] = operand_strides[static_cast<size_t>(operand_dim_index)];
  }
  return strides;
}

bool IsFloatGpuTensor(Ort::ConstValue value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  return info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
         IsGpuMemory(value.GetTensorMemoryInfo());
}

struct PowAffineSplitReduceOutput {
  size_t output_index;
  int64_t offset;
  int64_t width;
  MusaSplitReduceMode mode;
};

}  // namespace

struct PowAffineSplitReduceFusionCompute : FusionNodeCompute {
  PowAffineSplitReduceFusionCompute(
      size_t input_index, size_t exponent_index, size_t affine_index,
      std::optional<size_t> affine_output_index, MusaPowAffineOp affine_op,
      std::vector<PowAffineSplitReduceOutput> outputs)
      : input_index(input_index),
        exponent_index(exponent_index),
        affine_index(affine_index),
        affine_output_index(affine_output_index),
        affine_op(affine_op),
        outputs(std::move(outputs)) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      if (outputs.size() != 2) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "PowAffineSplitReduce requires two outputs");
      }

      Ort::ConstValue input = ctx.GetInput(input_index);
      Ort::ConstValue exponent = ctx.GetInput(exponent_index);
      Ort::ConstValue affine = ctx.GetInput(affine_index);
      if (!IsFloatGpuTensor(input) || !IsFloatGpuTensor(exponent) ||
          !IsFloatGpuTensor(affine)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "PowAffineSplitReduce requires MUSA float inputs");
      }

      std::vector<int64_t> input_shape =
          input.GetTensorTypeAndShapeInfo().GetShape();
      std::vector<int64_t> exponent_shape =
          exponent.GetTensorTypeAndShapeInfo().GetShape();
      std::vector<int64_t> affine_shape =
          affine.GetTensorTypeAndShapeInfo().GetShape();
      if (input_shape.size() != 3 || input_shape[1] <= 0 ||
          input_shape[2] <= 0) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "PowAffineSplitReduce requires rank-3 input");
      }

      const int64_t batch = input_shape[0];
      const int64_t axis_dim = input_shape[1];
      const int64_t inner = input_shape[2];
      MusaPowAffineSplitReduceParams params{};
      params.batch = batch;
      params.axis_dim = axis_dim;
      params.inner = inner;
      params.affine_op = affine_op;

      const auto exponent_strides =
          BroadcastStridesForRank3(exponent_shape, input_shape);
      const auto affine_strides =
          BroadcastStridesForRank3(affine_shape, input_shape);
      std::copy(exponent_strides.begin(), exponent_strides.end(),
                params.exponent_strides);
      std::copy(affine_strides.begin(), affine_strides.end(),
                params.affine_strides);

      float* affine_output_data = nullptr;
      if (affine_output_index.has_value()) {
        Ort::UnownedValue affine_output =
            ctx.GetOutput(*affine_output_index, {batch, axis_dim, inner});
        if (!IsGpuMemory(affine_output.GetTensorMemoryInfo())) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED,
              "PowAffineSplitReduce requires MUSA affine output");
        }
        affine_output_data = affine_output.GetTensorMutableData<float>();
      }

      std::array<float*, 2> output_data = {nullptr, nullptr};
      for (size_t i = 0; i < outputs.size(); ++i) {
        const PowAffineSplitReduceOutput& output = outputs[i];
        if (output.offset < 0 || output.width <= 0 ||
            output.offset + output.width > axis_dim) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "PowAffineSplitReduce output spec is invalid");
        }
        Ort::UnownedValue y = ctx.GetOutput(output.output_index, {batch, inner});
        if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED,
              "PowAffineSplitReduce requires MUSA outputs");
        }
        output_data[i] = y.GetTensorMutableData<float>();
      }

      params.offset0 = outputs[0].offset;
      params.width0 = outputs[0].width;
      params.mode0 = outputs[0].mode;
      params.offset1 = outputs[1].offset;
      params.width1 = outputs[1].width;
      params.mode1 = outputs[1].mode;
      return LaunchStatus(LaunchMusaPowAffineSplitReduce2Float(
          input.GetTensorData<float>(), exponent.GetTensorData<float>(),
          affine.GetTensorData<float>(), affine_output_data, output_data[0],
          output_data[1], params, nullptr));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t input_index;
  size_t exponent_index;
  size_t affine_index;
  std::optional<size_t> affine_output_index;
  MusaPowAffineOp affine_op;
  std::vector<PowAffineSplitReduceOutput> outputs;
};

bool IsPowAffineSplitReduceFusionGraph(Ort::ConstGraph graph) {
  int pow_count = 0;
  int affine_count = 0;
  int split_count = 0;
  int reduce_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Pow")) {
      ++pow_count;
    } else if (IsOnnxOp(node, "Add") || IsOnnxOp(node, "Sub")) {
      ++affine_count;
    } else if (IsOnnxOp(node, "Split")) {
      ++split_count;
    } else if (IsOnnxOp(node, "ReduceProd") || IsOnnxOp(node, "ReduceMean")) {
      ++reduce_count;
    } else {
      return false;
    }
  }
  return pow_count == 1 && affine_count == 1 && split_count == 1 &&
         reduce_count == 2;
}

std::unique_ptr<FusionNodeCompute> CreatePowAffineSplitReduceFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode split_node = FindSplitNode(graph);
  if (!split_node) {
    throw std::runtime_error("PowAffineSplitReduce requires Split");
  }
  if (ReadIntAttribute(split_node, "axis", 0) != 1) {
    throw std::runtime_error("PowAffineSplitReduce requires Split axis=1");
  }

  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  if (split_inputs.size() != 2 || split_outputs.size() != 2) {
    throw std::runtime_error("PowAffineSplitReduce requires two-way Split");
  }
  std::vector<int64_t> split_sizes = ReadIntInitializer(split_inputs[1]);
  if (split_sizes.size() != 2) {
    throw std::runtime_error("PowAffineSplitReduce requires two Split sizes");
  }

  Ort::ConstNode affine_node = ProducerNode(split_inputs[0]);
  if (!IsOnnxOp(affine_node, "Add") && !IsOnnxOp(affine_node, "Sub")) {
    throw std::runtime_error("PowAffineSplitReduce requires Add/Sub before Split");
  }
  std::vector<Ort::ConstValueInfo> affine_inputs = affine_node.GetInputs();
  std::vector<Ort::ConstValueInfo> affine_outputs = affine_node.GetOutputs();
  if (affine_inputs.size() != 2) {
    throw std::runtime_error("PowAffineSplitReduce requires binary Add/Sub");
  }
  if (affine_outputs.size() != 1) {
    throw std::runtime_error("PowAffineSplitReduce affine output mismatch");
  }

  int pow_input_index = -1;
  Ort::ConstNode pow_node{nullptr};
  for (size_t i = 0; i < affine_inputs.size(); ++i) {
    Ort::ConstNode producer = ProducerNode(affine_inputs[i]);
    if (IsOnnxOp(producer, "Pow")) {
      pow_input_index = static_cast<int>(i);
      pow_node = producer;
      break;
    }
  }
  if (!pow_node || (IsOnnxOp(affine_node, "Sub") && pow_input_index != 0)) {
    throw std::runtime_error(
        "PowAffineSplitReduce requires Pow + bias or Pow - bias");
  }

  std::vector<Ort::ConstValueInfo> pow_inputs = pow_node.GetInputs();
  if (pow_inputs.size() != 2) {
    throw std::runtime_error("PowAffineSplitReduce requires binary Pow");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  auto fused_output_indices = FusedOutputIndices(fused_node);
  const size_t input_index =
      GetMappedIndex(fused_input_indices, Name(pow_inputs[0]), "input");
  const size_t exponent_index =
      GetMappedIndex(fused_input_indices, Name(pow_inputs[1]), "exponent");
  const size_t affine_index = GetMappedIndex(
      fused_input_indices, Name(affine_inputs[static_cast<size_t>(1 - pow_input_index)]),
      "affine");
  std::optional<size_t> affine_output_index;
  auto affine_output_iter = fused_output_indices.find(Name(affine_outputs[0]));
  if (affine_output_iter != fused_output_indices.end()) {
    affine_output_index = affine_output_iter->second;
  }

  std::vector<PowAffineSplitReduceOutput> outputs;
  outputs.reserve(2);
  int64_t offset = 0;
  for (size_t i = 0; i < split_outputs.size(); ++i) {
    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
        split_outputs[i].GetConsumers();
    if (consumers.size() != 1 ||
        !(IsOnnxOp(consumers[0].node, "ReduceProd") ||
          IsOnnxOp(consumers[0].node, "ReduceMean")) ||
        !AxesInputIsAxis1(consumers[0].node) ||
        ReadIntAttribute(consumers[0].node, "keepdims", 1) != 0) {
      throw std::runtime_error(
          "PowAffineSplitReduce requires Reduce(axis=1, keepdims=0)");
    }
    std::vector<Ort::ConstValueInfo> reduce_outputs =
        consumers[0].node.GetOutputs();
    if (reduce_outputs.size() != 1) {
      throw std::runtime_error("PowAffineSplitReduce reduce output mismatch");
    }
    outputs.push_back(PowAffineSplitReduceOutput{
        GetMappedIndex(fused_output_indices, Name(reduce_outputs[0]), "output"),
        offset,
        split_sizes[i],
        ReduceModeForNode(consumers[0].node),
    });
    offset += split_sizes[i];
  }

  return std::make_unique<PowAffineSplitReduceFusionCompute>(
      input_index, exponent_index, affine_index, affine_output_index,
      IsOnnxOp(affine_node, "Sub") ? MusaPowAffineOp::Sub
                                   : MusaPowAffineOp::Add,
      std::move(outputs));
}
