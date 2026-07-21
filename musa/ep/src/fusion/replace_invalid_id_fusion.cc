// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "fusion/replace_invalid_id_fusion.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/graph_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/replace_invalid_id_impl.h"

namespace {

std::unordered_map<std::string, Ort::ConstNode> ProducersInGraph(
    Ort::ConstGraph graph) {
  std::unordered_map<std::string, Ort::ConstNode> producers;
  for (Ort::ConstNode node : graph.GetNodes()) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      producers.emplace(musa_ep::Name(output), node);
    }
  }
  return producers;
}

Ort::ConstNode ProducerInGraph(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo value_info) {
  auto it = producers.find(musa_ep::Name(value_info));
  return it == producers.end() ? Ort::ConstNode{nullptr} : it->second;
}

std::unordered_map<std::string, size_t> ValueIndices(
    const std::vector<Ort::ConstValueInfo>& values) {
  std::unordered_map<std::string, size_t> indices;
  for (size_t i = 0; i < values.size(); ++i) {
    indices.emplace(musa_ep::Name(values[i]), i);
  }
  return indices;
}

size_t GetMappedIndex(const std::unordered_map<std::string, size_t>& indices,
                      const std::string& name, const char* kind) {
  auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error(std::string("unable to map ReplaceInvalidId ") +
                             kind + " " + name);
  }
  return it->second;
}

int64_t ReadScalarIntInitializerOrThrow(Ort::ConstValueInfo value_info,
                                        ONNXTensorElementDataType expected_type,
                                        const char* kind) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    throw std::runtime_error(std::string("ReplaceInvalidId requires scalar ") +
                             kind + " initializer");
  }
  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error(std::string("unable to read ReplaceInvalidId ") +
                             kind + " initializer");
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementCount() != 1 || info.GetElementType() != expected_type) {
    throw std::runtime_error(std::string("ReplaceInvalidId ") + kind +
                             " must be a same-type scalar initializer");
  }
  if (expected_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return static_cast<int64_t>(value.GetTensorData<int32_t>()[0]);
  }
  if (expected_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return value.GetTensorData<int64_t>()[0];
  }
  throw std::runtime_error("ReplaceInvalidId supports int32/int64 only");
}

bool IsGpuTensor(Ort::ConstValue value) {
  return IsGpuMemory(value.GetTensorMemoryInfo());
}

struct ReplaceInvalidIdFusionCompute : FusionNodeCompute {
  ReplaceInvalidIdFusionCompute(size_t input_index, size_t output_index,
                                ONNXTensorElementDataType element_type,
                                int64_t threshold, int64_t replacement)
      : input_index(input_index),
        output_index(output_index),
        element_type(element_type),
        threshold(threshold),
        replacement(replacement) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue input = ctx.GetInput(input_index);
      if (!IsGpuTensor(input)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "ReplaceInvalidId requires a MUSA input tensor");
      }

      auto input_info = input.GetTensorTypeAndShapeInfo();
      if (input_info.GetElementType() != element_type) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "ReplaceInvalidId input dtype changed");
      }
      std::vector<int64_t> output_shape = input_info.GetShape();
      const int64_t count = input_info.GetElementCount();
      Ort::UnownedValue output = ctx.GetOutput(output_index, output_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo()) ||
          output.GetTensorTypeAndShapeInfo().GetElementType() != element_type) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "ReplaceInvalidId requires a same-type MUSA output tensor");
      }

      if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        return LaunchStatus(LaunchMusaReplaceInvalidIdInt32Kernel(
            input.GetTensorData<int32_t>(),
            output.GetTensorMutableData<int32_t>(), count,
            static_cast<int32_t>(threshold), static_cast<int32_t>(replacement),
            GetComputeStream(ctx)));
      }
      if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        return LaunchStatus(LaunchMusaReplaceInvalidIdInt64Kernel(
            input.GetTensorData<int64_t>(),
            output.GetTensorMutableData<int64_t>(), count, threshold,
            replacement, GetComputeStream(ctx)));
      }
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "ReplaceInvalidId supports int32/int64 only");
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t input_index;
  size_t output_index;
  ONNXTensorElementDataType element_type;
  int64_t threshold;
  int64_t replacement;
};

}  // namespace

bool IsReplaceInvalidIdFusionGraph(Ort::ConstGraph graph) {
  int less_equal_count = 0;
  int not_count = 0;
  int cast_count = 0;
  int mul_count = 0;
  int add_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (musa_ep::IsOnnxOp(node, "LessOrEqual")) {
      ++less_equal_count;
    } else if (musa_ep::IsOnnxOp(node, "Not")) {
      ++not_count;
    } else if (musa_ep::IsOnnxOp(node, "Cast")) {
      ++cast_count;
    } else if (musa_ep::IsOnnxOp(node, "Mul")) {
      ++mul_count;
    } else if (musa_ep::IsOnnxOp(node, "Add")) {
      ++add_count;
    } else {
      return false;
    }
  }
  return less_equal_count == 1 && not_count == 1 && cast_count == 2 &&
         mul_count == 2 && add_count == 1;
}

std::unique_ptr<FusionNodeCompute> CreateReplaceInvalidIdFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  auto producers = ProducersInGraph(graph);
  auto input_indices = ValueIndices(fused_node.GetInputs());
  auto output_indices = ValueIndices(fused_node.GetOutputs());

  Ort::ConstNode less_equal_node{nullptr};
  Ort::ConstNode add_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (musa_ep::IsOnnxOp(node, "LessOrEqual")) {
      less_equal_node = node;
    } else if (musa_ep::IsOnnxOp(node, "Add")) {
      add_node = node;
    }
  }
  if (!less_equal_node || !add_node) {
    throw std::runtime_error("ReplaceInvalidId requires LessOrEqual and Add");
  }

  std::vector<Ort::ConstValueInfo> less_equal_inputs =
      less_equal_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  if (less_equal_inputs.size() != 2 || add_outputs.size() != 1) {
    throw std::runtime_error("invalid ReplaceInvalidId graph boundary");
  }
  Ort::ConstValueInfo input = less_equal_inputs[0];
  Ort::ConstValueInfo threshold_input = less_equal_inputs[1];
  auto input_info = input.TypeInfo().GetTensorTypeAndShapeInfo();
  const ONNXTensorElementDataType element_type = input_info.GetElementType();

  Ort::ConstValueInfo replacement_input{nullptr};
  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  for (Ort::ConstValueInfo add_input : add_inputs) {
    Ort::ConstNode mul_node = ProducerInGraph(producers, add_input);
    if (!musa_ep::IsOnnxOp(mul_node, "Mul")) {
      continue;
    }
    for (Ort::ConstValueInfo mul_input : mul_node.GetInputs()) {
      if (mul_input.IsConstantInitializer()) {
        replacement_input = mul_input;
      }
    }
  }
  if (replacement_input == nullptr) {
    throw std::runtime_error(
        "ReplaceInvalidId requires a replacement initializer");
  }

  const int64_t threshold = ReadScalarIntInitializerOrThrow(
      threshold_input, element_type, "threshold");
  const int64_t replacement = ReadScalarIntInitializerOrThrow(
      replacement_input, element_type, "replacement");
  return std::make_unique<ReplaceInvalidIdFusionCompute>(
      GetMappedIndex(input_indices, musa_ep::Name(input), "input"),
      GetMappedIndex(output_indices, musa_ep::Name(add_outputs[0]), "output"),
      element_type, threshold, replacement);
}
