// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fusion/math_concat_log_fusion.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernels/math/math_concat_log_impl.h"
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
    throw std::runtime_error(std::string("unable to map MathConcatLog ") +
                             kind + " " + name);
  }
  return it->second;
}

bool IsConstantInitializerValueInfo(Ort::ConstValueInfo value_info) {
  return value_info != nullptr && value_info.IsConstantInitializer();
}

float ReadScalarFloatInitializer(Ort::ConstValueInfo value_info,
                                 const char* kind) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    throw std::runtime_error(std::string("MathConcatLog requires scalar ") +
                             kind + " initializer");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error(std::string("unable to read MathConcatLog ") +
                             kind + " initializer");
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      info.GetElementCount() != 1) {
    throw std::runtime_error(std::string("MathConcatLog ") + kind +
                             " initializer must be scalar float");
  }
  return value.GetTensorData<float>()[0];
}

bool IsFloatGpuTensor(Ort::ConstValue value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  return info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
         IsGpuMemory(value.GetTensorMemoryInfo());
}

}  // namespace

struct MathConcatLogFusionCompute : FusionNodeCompute {
  MathConcatLogFusionCompute(size_t input_index, size_t output_index,
                             float max_value, float add_value,
                             float scale_value)
      : input_index(input_index),
        output_index(output_index),
        max_value(max_value),
        add_value(add_value),
        scale_value(scale_value) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue input = ctx.GetInput(input_index);
      if (!IsFloatGpuTensor(input)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "MathConcatLog requires a MUSA float input");
      }

      auto input_info = input.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> output_shape = input_info.GetShape();
      Ort::UnownedValue output = ctx.GetOutput(output_index, output_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "MathConcatLog requires a MUSA output");
      }

      return LaunchStatus(LaunchMusaMathConcatLogKernel(
          input.GetTensorData<float>(), output.GetTensorMutableData<float>(),
          input_info.GetElementCount(), max_value, add_value, scale_value,
          GetComputeStream(ctx)));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t input_index;
  size_t output_index;
  float max_value;
  float add_value;
  float scale_value;
};

bool IsMathConcatLogFusionGraph(Ort::ConstGraph graph) {
  int max_count = 0;
  int add_count = 0;
  int log_count = 0;
  int mul_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Max")) {
      ++max_count;
    } else if (IsOnnxOp(node, "Add")) {
      ++add_count;
    } else if (IsOnnxOp(node, "Log")) {
      ++log_count;
    } else if (IsOnnxOp(node, "Mul")) {
      ++mul_count;
    } else {
      return false;
    }
  }
  return max_count == 1 && add_count == 1 && log_count == 1 && mul_count == 1;
}

std::unique_ptr<FusionNodeCompute> CreateMathConcatLogFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  auto producers = ProducersInGraph(graph);
  auto fused_output_indices = FusedOutputIndices(fused_node);
  Ort::ConstNode mul_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Mul")) {
      std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
      if (outputs.size() == 1 &&
          fused_output_indices.count(Name(outputs[0])) != 0) {
        mul_node = node;
        break;
      }
    }
  }
  if (!mul_node) {
    throw std::runtime_error("MathConcatLog requires final Mul");
  }

  std::vector<Ort::ConstValueInfo> mul_inputs = mul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> mul_outputs = mul_node.GetOutputs();
  if (mul_inputs.size() != 2 || mul_outputs.size() != 1) {
    throw std::runtime_error("MathConcatLog final Mul is invalid");
  }

  Ort::ConstNode log_node{nullptr};
  Ort::ConstValueInfo scale_input{nullptr};
  for (Ort::ConstValueInfo input : mul_inputs) {
    if (IsConstantInitializerValueInfo(input)) {
      scale_input = input;
    } else {
      Ort::ConstNode producer = ProducerInGraph(producers, input);
      if (IsOnnxOp(producer, "Log")) {
        log_node = producer;
      } else {
        scale_input = input;
      }
    }
  }
  if (!IsOnnxOp(log_node, "Log")) {
    throw std::runtime_error("MathConcatLog requires Log before Mul");
  }

  std::vector<Ort::ConstValueInfo> log_inputs = log_node.GetInputs();
  if (log_inputs.size() != 1) {
    throw std::runtime_error("MathConcatLog Log is invalid");
  }
  Ort::ConstNode add_node = ProducerInGraph(producers, log_inputs[0]);
  if (!IsOnnxOp(add_node, "Add")) {
    throw std::runtime_error("MathConcatLog requires Add before Log");
  }

  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  if (add_inputs.size() != 2) {
    throw std::runtime_error("MathConcatLog Add is invalid");
  }
  Ort::ConstNode max_node{nullptr};
  Ort::ConstValueInfo add_input{nullptr};
  for (Ort::ConstValueInfo input : add_inputs) {
    if (IsConstantInitializerValueInfo(input)) {
      add_input = input;
    } else {
      Ort::ConstNode producer = ProducerInGraph(producers, input);
      if (IsOnnxOp(producer, "Max")) {
        max_node = producer;
      } else {
        add_input = input;
      }
    }
  }
  if (!IsOnnxOp(max_node, "Max")) {
    throw std::runtime_error("MathConcatLog requires Max before Add");
  }

  std::vector<Ort::ConstValueInfo> max_inputs = max_node.GetInputs();
  if (max_inputs.size() != 2) {
    throw std::runtime_error("MathConcatLog Max is invalid");
  }
  Ort::ConstValueInfo input{nullptr};
  Ort::ConstValueInfo max_input{nullptr};
  for (Ort::ConstValueInfo candidate : max_inputs) {
    if (IsConstantInitializerValueInfo(candidate)) {
      max_input = candidate;
    } else {
      input = candidate;
    }
  }
  if (input == nullptr || max_input == nullptr) {
    throw std::runtime_error("MathConcatLog Max requires input and constant");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  return std::make_unique<MathConcatLogFusionCompute>(
      GetMappedIndex(fused_input_indices, Name(input), "input"),
      GetMappedIndex(fused_output_indices, Name(mul_outputs[0]), "output"),
      ReadScalarFloatInitializer(max_input, "max"),
      ReadScalarFloatInitializer(add_input, "add"),
      ReadScalarFloatInitializer(scale_input, "scale"));
}
