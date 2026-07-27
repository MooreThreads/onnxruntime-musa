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

#include "fusion/rms_norm_fusion.h"

#include <climits>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/nn/rms_norm_impl.h"
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
    throw std::runtime_error(std::string("unable to map RmsNorm ") + kind +
                             " " + name);
  }
  return it->second;
}

float ReadScalarFloat(Ort::ConstValue value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementCount() != 1 ||
      info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("RmsNorm epsilon input must be scalar float");
  }

  std::vector<uint8_t> bytes;
  Ort::ThrowOnError(CopyToHost(value, bytes));
  float result = 0.0f;
  std::memcpy(&result, bytes.data(), sizeof(float));
  return result;
}

bool IsFloatGpuTensor(Ort::ConstValue value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType elem_type = info.GetElementType();
  return (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
          elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 ||
          elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) &&
         IsGpuMemory(value.GetTensorMemoryInfo());
}

bool IsFloatTensor(Ort::ConstValue value) {
  return value.GetTensorTypeAndShapeInfo().GetElementType() ==
         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

}  // namespace

struct RmsNormFusionCompute : FusionNodeCompute {
  RmsNormFusionCompute(size_t input_index, size_t epsilon_index,
                       size_t gamma_index, size_t output_index)
      : input_index(input_index),
        epsilon_index(epsilon_index),
        gamma_index(gamma_index),
        output_index(output_index) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue input = ctx.GetInput(input_index);
      Ort::ConstValue epsilon = ctx.GetInput(epsilon_index);
      Ort::ConstValue gamma = ctx.GetInput(gamma_index);
      if (!IsFloatGpuTensor(input)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "RmsNorm requires a MUSA floating input");
      }
      if (!IsFloatTensor(gamma)) {
        return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                          "RmsNorm requires float gamma");
      }

      auto input_info = input.GetTensorTypeAndShapeInfo();
      auto gamma_info = gamma.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> input_shape = input_info.GetShape();
      std::vector<int64_t> gamma_shape = gamma_info.GetShape();
      if (input_shape.size() < 2 || input_shape.back() <= 0 ||
          gamma_shape.size() != 1 || gamma_shape[0] != input_shape.back()) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "RmsNorm requires rank >= 2 input and 1-D gamma matching last dim");
      }

      int64_t rows = 1;
      for (size_t i = 0; i + 1 < input_shape.size(); ++i) {
        if (input_shape[i] < 0) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "RmsNorm requires concrete dimensions");
        }
        rows *= input_shape[i];
      }
      const int64_t norm_size = input_shape.back();
      if (rows > INT32_MAX || norm_size > INT32_MAX) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "RmsNorm shape exceeds MUSA kernel limits");
      }

      MusaElementType elem_type;
      if (!ToMusaElementType(input_info.GetElementType(), elem_type)) {
        return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                          "RmsNorm unsupported dtype");
      }

      Ort::UnownedValue output = ctx.GetOutput(output_index, input_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                          "RmsNorm requires MUSA output");
      }

      const musaStream_t stream = GetComputeStream(ctx);
      DeviceInputBuffer gamma_buffer;
      RETURN_IF_ERROR(gamma_buffer.Bind(gamma, stream));
      return LaunchStatus(LaunchMusaRmsNormKernel(
          input.GetTensorRawData(),
          reinterpret_cast<const float*>(gamma_buffer.data()),
          output.GetTensorMutableRawData(), rows, norm_size,
          ReadScalarFloat(epsilon), elem_type, stream));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t input_index;
  size_t epsilon_index;
  size_t gamma_index;
  size_t output_index;
};

bool IsRmsNormFusionGraph(Ort::ConstGraph graph) {
  int mul_count = 0;
  int reduce_mean_count = 0;
  int add_count = 0;
  int sqrt_count = 0;
  int div_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Mul")) {
      ++mul_count;
    } else if (IsOnnxOp(node, "ReduceMean")) {
      ++reduce_mean_count;
    } else if (IsOnnxOp(node, "Add")) {
      ++add_count;
    } else if (IsOnnxOp(node, "Sqrt")) {
      ++sqrt_count;
    } else if (IsOnnxOp(node, "Div")) {
      ++div_count;
    } else {
      return false;
    }
  }
  return mul_count == 2 && reduce_mean_count == 1 && add_count == 1 &&
         sqrt_count == 1 && div_count == 1;
}

std::unique_ptr<FusionNodeCompute> CreateRmsNormFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  auto producers = ProducersInGraph(graph);
  auto fused_output_indices = FusedOutputIndices(fused_node);
  Ort::ConstNode output_mul_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Mul")) {
      std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
      if (outputs.size() == 1 &&
          fused_output_indices.count(Name(outputs[0])) != 0) {
        output_mul_node = node;
      }
    }
  }
  if (!output_mul_node) {
    throw std::runtime_error("RmsNorm requires final Mul");
  }

  std::vector<Ort::ConstValueInfo> output_mul_inputs =
      output_mul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> output_mul_outputs =
      output_mul_node.GetOutputs();
  if (output_mul_inputs.size() != 2 || output_mul_outputs.size() != 1) {
    throw std::runtime_error("RmsNorm final Mul is invalid");
  }

  Ort::ConstNode div_node = ProducerInGraph(producers, output_mul_inputs[0]);
  Ort::ConstValueInfo gamma_input = output_mul_inputs[1];
  if (!IsOnnxOp(div_node, "Div")) {
    div_node = ProducerInGraph(producers, output_mul_inputs[1]);
    gamma_input = output_mul_inputs[0];
  }
  if (!IsOnnxOp(div_node, "Div")) {
    throw std::runtime_error("RmsNorm requires Div before final Mul");
  }

  std::vector<Ort::ConstValueInfo> div_inputs = div_node.GetInputs();
  if (div_inputs.size() != 2) {
    throw std::runtime_error("RmsNorm requires binary Div");
  }
  Ort::ConstValueInfo input = div_inputs[0];
  Ort::ConstNode sqrt_node = ProducerInGraph(producers, div_inputs[1]);
  if (!IsOnnxOp(sqrt_node, "Sqrt")) {
    throw std::runtime_error("RmsNorm requires Sqrt denominator");
  }

  std::vector<Ort::ConstValueInfo> sqrt_inputs = sqrt_node.GetInputs();
  if (sqrt_inputs.size() != 1) {
    throw std::runtime_error("RmsNorm Sqrt is invalid");
  }
  Ort::ConstNode add_node = ProducerInGraph(producers, sqrt_inputs[0]);
  if (!IsOnnxOp(add_node, "Add")) {
    throw std::runtime_error("RmsNorm requires Add before Sqrt");
  }

  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  if (add_inputs.size() != 2) {
    throw std::runtime_error("RmsNorm Add is invalid");
  }
  Ort::ConstNode reduce_node = ProducerInGraph(producers, add_inputs[0]);
  Ort::ConstValueInfo epsilon_input = add_inputs[1];
  if (!IsOnnxOp(reduce_node, "ReduceMean")) {
    reduce_node = ProducerInGraph(producers, add_inputs[1]);
    epsilon_input = add_inputs[0];
  }
  if (!IsOnnxOp(reduce_node, "ReduceMean")) {
    throw std::runtime_error("RmsNorm requires ReduceMean before Add");
  }

  std::vector<Ort::ConstValueInfo> reduce_inputs = reduce_node.GetInputs();
  if (reduce_inputs.empty()) {
    throw std::runtime_error("RmsNorm ReduceMean is invalid");
  }
  Ort::ConstNode square_node = ProducerInGraph(producers, reduce_inputs[0]);
  if (!IsOnnxOp(square_node, "Mul")) {
    throw std::runtime_error("RmsNorm requires Mul(x, x) square");
  }
  std::vector<Ort::ConstValueInfo> square_inputs = square_node.GetInputs();
  if (square_inputs.size() != 2 || Name(square_inputs[0]) != Name(input) ||
      Name(square_inputs[1]) != Name(input)) {
    throw std::runtime_error("RmsNorm square must be Mul(input, input)");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  return std::make_unique<RmsNormFusionCompute>(
      GetMappedIndex(fused_input_indices, Name(input), "input"),
      GetMappedIndex(fused_input_indices, Name(epsilon_input), "epsilon"),
      GetMappedIndex(fused_input_indices, Name(gamma_input), "gamma"),
      GetMappedIndex(fused_output_indices, Name(output_mul_outputs[0]),
                     "output"));
}
