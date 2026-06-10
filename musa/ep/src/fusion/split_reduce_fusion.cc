// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/split_reduce_fusion.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/reduction/split_reduce_impl.h"
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
  if (!value_info.IsConstantInitializer()) {
    throw std::runtime_error("SplitReduce requires constant Split sizes");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("SplitReduce failed to read Split initializer");
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
  throw std::runtime_error("SplitReduce requires int32/int64 Split sizes");
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
  throw std::runtime_error("SplitReduce unsupported reduce op");
}

struct SplitReduceOutput {
  size_t output_index;
  int64_t offset;
  int64_t width;
  MusaSplitReduceMode mode;
};

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
    throw std::runtime_error("unable to map SplitReduce output " + output_name);
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

}  // namespace

struct SplitReduceFusionCompute : FusionNodeCompute {
  explicit SplitReduceFusionCompute(std::vector<SplitReduceOutput> outputs)
      : outputs(std::move(outputs)) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      if (outputs.size() != 2) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "SplitReduce requires two outputs");
      }

      Ort::ConstValue input = ctx.GetInput(0);
      auto input_info = input.GetTensorTypeAndShapeInfo();
      if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          !IsGpuMemory(input.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "SplitReduce requires MUSA float input");
      }
      std::vector<int64_t> input_shape = input_info.GetShape();
      if (input_shape.size() != 3 || input_shape[1] <= 0 ||
          input_shape[2] <= 0) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "SplitReduce requires rank-3 input");
      }

      const int64_t batch = input_shape[0];
      const int64_t axis_dim = input_shape[1];
      const int64_t inner = input_shape[2];
      std::vector<float*> output_data(outputs.size(), nullptr);
      for (const SplitReduceOutput& output : outputs) {
        if (output.offset < 0 || output.width <= 0 ||
            output.offset + output.width > axis_dim ||
            output.output_index >= outputs.size()) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT, "SplitReduce output spec is invalid");
        }
        Ort::UnownedValue y = ctx.GetOutput(output.output_index, {batch, inner});
        if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "SplitReduce requires MUSA outputs");
        }
        output_data[output.output_index] = y.GetTensorMutableData<float>();
      }

      return LaunchStatus(LaunchMusaSplitReduce2Float(
          input.GetTensorData<float>(), output_data[0], output_data[1], batch,
          axis_dim, inner, outputs[0].offset, outputs[0].width,
          outputs[0].mode, outputs[1].offset, outputs[1].width,
          outputs[1].mode, nullptr));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  std::vector<SplitReduceOutput> outputs;
};

bool IsSplitReduceFusionGraph(Ort::ConstGraph graph) {
  int split_count = 0;
  int reduce_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Split")) {
      ++split_count;
    } else if (IsOnnxOp(node, "ReduceProd") || IsOnnxOp(node, "ReduceMean")) {
      ++reduce_count;
    } else {
      return false;
    }
  }
  return split_count == 1 && reduce_count == 2;
}

std::unique_ptr<FusionNodeCompute> CreateSplitReduceFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode split_node = FindSplitNode(graph);
  if (!split_node) {
    throw std::runtime_error("SplitReduce requires Split");
  }
  if (ReadIntAttribute(split_node, "axis", 0) != 1) {
    throw std::runtime_error("SplitReduce requires Split axis=1");
  }

  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  if (split_inputs.size() != 2 || split_outputs.size() != 2) {
    throw std::runtime_error("SplitReduce requires two-way Split");
  }
  std::vector<int64_t> split_sizes = ReadIntInitializer(split_inputs[1]);
  if (split_sizes.size() != 2) {
    throw std::runtime_error("SplitReduce requires two Split sizes");
  }

  auto fused_output_indices = FusedOutputIndices(fused_node);
  std::vector<SplitReduceOutput> outputs;
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
      throw std::runtime_error("SplitReduce requires Reduce(axis=1, keepdims=0)");
    }
    std::vector<Ort::ConstValueInfo> reduce_outputs =
        consumers[0].node.GetOutputs();
    if (reduce_outputs.size() != 1) {
      throw std::runtime_error("SplitReduce reduce output mismatch");
    }
    outputs.push_back(SplitReduceOutput{
        GetFusedOutputIndex(fused_output_indices, Name(reduce_outputs[0])),
        offset,
        split_sizes[i],
        ReduceModeForNode(consumers[0].node),
    });
    offset += split_sizes[i];
  }
  return std::make_unique<SplitReduceFusionCompute>(std::move(outputs));
}
