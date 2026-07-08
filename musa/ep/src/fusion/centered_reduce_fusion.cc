// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "fusion/centered_reduce_fusion.h"

#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/reduction/centered_reduce_impl.h"
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

std::unordered_map<std::string, Ort::ConstNode> ProducersInGraph(
    Ort::ConstGraph graph) {
  std::unordered_map<std::string, Ort::ConstNode> producers;
  for (Ort::ConstNode node : graph.GetNodes()) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      producers.emplace(Name(output), node);
    }
  }
  return producers;
}

Ort::ConstNode ProducerInGraph(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo value_info) {
  auto it = producers.find(Name(value_info));
  return it == producers.end() ? Ort::ConstNode{nullptr} : it->second;
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
    throw std::runtime_error(std::string("unable to map CenteredReduce ") +
                             kind + " " + name);
  }
  return it->second;
}

MusaReduceOp ReduceOpForNode(Ort::ConstNode node) {
  if (IsOnnxOp(node, "ReduceSum")) {
    return MusaReduceOp::Sum;
  }
  if (IsOnnxOp(node, "ReduceProd")) {
    return MusaReduceOp::Prod;
  }
  throw std::runtime_error("CenteredReduce unsupported reduce op");
}

Ort::ConstNode FindSecondReduceNode(
    Ort::ConstGraph graph,
    const std::unordered_map<std::string, Ort::ConstNode>& producers) {
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, "ReduceSum") && !IsOnnxOp(node, "ReduceProd")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    if (inputs.empty()) {
      continue;
    }
    Ort::ConstNode mul_node = ProducerInGraph(producers, inputs[0]);
    if (IsOnnxOp(mul_node, "Mul")) {
      return node;
    }
  }
  return Ort::ConstNode{nullptr};
}

bool IsFloatGpuTensor(Ort::ConstValue value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  return info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
         IsGpuMemory(value.GetTensorMemoryInfo());
}

}  // namespace

struct CenteredReduceFusionCompute : FusionNodeCompute {
  CenteredReduceFusionCompute(size_t input_index, size_t first_output_index,
                              size_t second_output_index, MusaReduceOp first_op,
                              MusaReduceOp second_op)
      : input_index(input_index),
        first_output_index(first_output_index),
        second_output_index(second_output_index),
        first_op(first_op),
        second_op(second_op) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue input = ctx.GetInput(input_index);
      if (!IsFloatGpuTensor(input)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "CenteredReduce requires a MUSA float input");
      }

      std::vector<int64_t> input_shape =
          input.GetTensorTypeAndShapeInfo().GetShape();
      if (input_shape.size() < 2 || input_shape.back() <= 0) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "CenteredReduce requires rank >= 2 and static last dim");
      }

      int64_t rows = 1;
      for (size_t i = 0; i + 1 < input_shape.size(); ++i) {
        if (input_shape[i] < 0) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED,
              "CenteredReduce requires concrete runtime dimensions");
        }
        rows *= input_shape[i];
      }
      const int64_t inner = input_shape.back();

      std::vector<int64_t> output_shape = input_shape;
      output_shape.back() = 1;
      Ort::UnownedValue first_output =
          ctx.GetOutput(first_output_index, output_shape);
      Ort::UnownedValue second_output =
          ctx.GetOutput(second_output_index, output_shape);
      if (!IsGpuMemory(first_output.GetTensorMemoryInfo()) ||
          !IsGpuMemory(second_output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "CenteredReduce requires MUSA outputs");
      }

      return LaunchStatus(LaunchMusaCenteredReduceFloatKernel(
          input.GetTensorData<float>(),
          first_output.GetTensorMutableData<float>(),
          second_output.GetTensorMutableData<float>(), rows, inner, first_op,
          second_op, GetComputeStream(ctx)));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t input_index;
  size_t first_output_index;
  size_t second_output_index;
  MusaReduceOp first_op;
  MusaReduceOp second_op;
};

bool IsCenteredReduceFusionGraph(Ort::ConstGraph graph) {
  int reduce_count = 0;
  int sub_count = 0;
  int mul_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "ReduceSum") || IsOnnxOp(node, "ReduceProd")) {
      ++reduce_count;
    } else if (IsOnnxOp(node, "Sub")) {
      ++sub_count;
    } else if (IsOnnxOp(node, "Mul")) {
      ++mul_count;
    } else {
      return false;
    }
  }
  return reduce_count == 2 && sub_count == 1 && mul_count == 1;
}

std::unique_ptr<FusionNodeCompute> CreateCenteredReduceFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  auto producers = ProducersInGraph(graph);
  Ort::ConstNode second_reduce_node = FindSecondReduceNode(graph, producers);
  if (!second_reduce_node) {
    throw std::runtime_error("CenteredReduce requires trailing Reduce");
  }

  std::vector<Ort::ConstValueInfo> second_reduce_inputs =
      second_reduce_node.GetInputs();
  std::vector<Ort::ConstValueInfo> second_reduce_outputs =
      second_reduce_node.GetOutputs();
  if (second_reduce_inputs.empty() || second_reduce_outputs.size() != 1) {
    throw std::runtime_error("CenteredReduce trailing Reduce is invalid");
  }

  Ort::ConstNode mul_node = ProducerInGraph(producers, second_reduce_inputs[0]);
  std::vector<Ort::ConstValueInfo> mul_inputs = mul_node.GetInputs();
  if (mul_inputs.size() != 2 || Name(mul_inputs[0]) != Name(mul_inputs[1])) {
    throw std::runtime_error("CenteredReduce requires Mul(x, x)");
  }

  Ort::ConstNode sub_node = ProducerInGraph(producers, mul_inputs[0]);
  if (!IsOnnxOp(sub_node, "Sub")) {
    throw std::runtime_error("CenteredReduce requires Sub before Mul");
  }
  std::vector<Ort::ConstValueInfo> sub_inputs = sub_node.GetInputs();
  if (sub_inputs.size() != 2) {
    throw std::runtime_error("CenteredReduce requires binary Sub");
  }

  Ort::ConstNode first_reduce_node = ProducerInGraph(producers, sub_inputs[1]);
  if (!IsOnnxOp(first_reduce_node, "ReduceSum") &&
      !IsOnnxOp(first_reduce_node, "ReduceProd")) {
    throw std::runtime_error("CenteredReduce requires first Reduce");
  }
  std::vector<Ort::ConstValueInfo> first_reduce_inputs =
      first_reduce_node.GetInputs();
  std::vector<Ort::ConstValueInfo> first_reduce_outputs =
      first_reduce_node.GetOutputs();
  if (first_reduce_inputs.empty() || first_reduce_outputs.size() != 1 ||
      Name(first_reduce_inputs[0]) != Name(sub_inputs[0])) {
    throw std::runtime_error(
        "CenteredReduce requires Sub(input, reduce(input))");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  auto fused_output_indices = FusedOutputIndices(fused_node);
  return std::make_unique<CenteredReduceFusionCompute>(
      GetMappedIndex(fused_input_indices, Name(first_reduce_inputs[0]),
                     "input"),
      GetMappedIndex(fused_output_indices, Name(first_reduce_outputs[0]),
                     "first output"),
      GetMappedIndex(fused_output_indices, Name(second_reduce_outputs[0]),
                     "second output"),
      ReduceOpForNode(first_reduce_node), ReduceOpForNode(second_reduce_node));
}
