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

#include "fusion/bucketize_gather_fusion.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph/graph_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/bucketize_gather_impl.h"

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

std::unordered_map<std::string, size_t> FusedInputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> fused_input_indices;
  std::vector<Ort::ConstValueInfo> fused_inputs = fused_node.GetInputs();
  for (size_t i = 0; i < fused_inputs.size(); ++i) {
    fused_input_indices.emplace(musa_ep::Name(fused_inputs[i]), i);
  }
  return fused_input_indices;
}

std::unordered_map<std::string, size_t> FusedOutputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> fused_output_indices;
  std::vector<Ort::ConstValueInfo> fused_outputs = fused_node.GetOutputs();
  for (size_t i = 0; i < fused_outputs.size(); ++i) {
    fused_output_indices.emplace(musa_ep::Name(fused_outputs[i]), i);
  }
  return fused_output_indices;
}

size_t GetMappedIndex(const std::unordered_map<std::string, size_t>& indices,
                      const std::string& name, const char* kind) {
  auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error(std::string("unable to map ") + kind + " " + name);
  }
  return it->second;
}

int64_t ReadRequiredScalarIntInitializer(Ort::ConstValueInfo value_info,
                                         const char* name) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    throw std::runtime_error(std::string(name) +
                             " must be a constant initializer");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error(std::string("failed to read ") + name);
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementCount() != 1) {
    throw std::runtime_error(std::string(name) +
                             " must be scalar or single-element tensor");
  }
  ONNXTensorElementDataType elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return value.GetTensorData<int64_t>()[0];
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return static_cast<int64_t>(value.GetTensorData<int32_t>()[0]);
  }
  throw std::runtime_error(std::string(name) + " must be int32/int64");
}

float ReadRequiredScalarFloatInitializer(Ort::ConstValueInfo value_info,
                                         const char* name) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    throw std::runtime_error(std::string(name) +
                             " must be a constant initializer");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error(std::string("failed to read ") + name);
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementCount() != 1 ||
      info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(std::string(name) +
                             " must be scalar or single-element float tensor");
  }
  return value.GetTensorData<float>()[0];
}

std::vector<int64_t> SqueezedGatherOutputShape(
    const std::vector<int64_t>& index_shape,
    const std::vector<int64_t>& table_shape, int64_t squeeze_axis) {
  std::vector<int64_t> gather_shape = index_shape;
  gather_shape.insert(gather_shape.end(), table_shape.begin() + 1,
                      table_shape.end());
  if (squeeze_axis < 0 ||
      squeeze_axis >= static_cast<int64_t>(gather_shape.size())) {
    throw std::runtime_error("invalid squeeze axis");
  }
  if (gather_shape[static_cast<size_t>(squeeze_axis)] != 1) {
    throw std::runtime_error("squeezed dimension must be 1");
  }
  gather_shape.erase(gather_shape.begin() + squeeze_axis);
  return gather_shape;
}

struct BucketizeGatherFusionCompute : FusionNodeCompute {
  BucketizeGatherFusionCompute(size_t table_index, size_t input_index,
                               int64_t modulus, int64_t offset,
                               float greater_threshold, int64_t squeeze_axis,
                               size_t output_index)
      : table_index(table_index),
        input_index(input_index),
        modulus(modulus),
        offset(offset),
        greater_threshold(greater_threshold),
        squeeze_axis(squeeze_axis),
        output_index(output_index) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue table = ctx.GetInput(table_index);
      Ort::ConstValue input = ctx.GetInput(input_index);

      if (!IsGpuMemory(table.GetTensorMemoryInfo()) ||
          !IsGpuMemory(input.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "BucketizeGather requires MUSA input tensors");
      }

      auto table_info = table.GetTensorTypeAndShapeInfo();
      auto input_info = input.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> table_shape = table_info.GetShape();
      std::vector<int64_t> input_shape = input_info.GetShape();
      if (table_shape.empty() || table_shape[0] <= 0) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "BucketizeGather table must have rows");
      }

      const size_t elem_size = ElementSize(table_info.GetElementType());
      if (elem_size == 0) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "BucketizeGather unsupported table dtype");
      }
      const size_t index_elem_size = ElementSize(input_info.GetElementType());
      if (index_elem_size != sizeof(int32_t) &&
          index_elem_size != sizeof(int64_t)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "BucketizeGather unsupported index dtype");
      }
      if (modulus <= 0 || offset < 0 || offset + modulus > table_shape[0]) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "BucketizeGather invalid modulus/offset");
      }

      std::vector<int64_t> out_shape =
          SqueezedGatherOutputShape(input_shape, table_shape, squeeze_axis);
      Ort::UnownedValue output = ctx.GetOutput(output_index, out_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "BucketizeGather requires MUSA output");
      }

      int64_t block_size = 1;
      for (size_t i = 1; i < table_shape.size(); ++i) {
        block_size *= table_shape[i];
      }
      const int64_t indices_count = NumElements(input_shape);

      DeviceInputBuffer table_buffer;
      DeviceInputBuffer input_buffer;
      RETURN_IF_ERROR(table_buffer.Bind(table, GetComputeStream(ctx)));
      RETURN_IF_ERROR(input_buffer.Bind(input, GetComputeStream(ctx)));
      return LaunchStatus(LaunchMusaBucketizeGatherKernel(
          table_buffer.data(), input_buffer.data(),
          output.GetTensorMutableRawData(), static_cast<int32_t>(elem_size),
          static_cast<int32_t>(index_elem_size), indices_count, table_shape[0],
          block_size, modulus, offset, greater_threshold,
          GetComputeStream(ctx)));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t table_index;
  size_t input_index;
  int64_t modulus;
  int64_t offset;
  float greater_threshold;
  int64_t squeeze_axis;
  size_t output_index;
};

}  // namespace

using musa_ep::IsOnnxOp;
using musa_ep::Name;
using musa_ep::ReadSmallIntInitializer;

bool IsBucketizeGatherFusionGraph(Ort::ConstGraph graph) {
  bool has_gather = false;
  bool has_greater = false;
  bool has_squeeze = false;
  bool has_sub = false;
  for (Ort::ConstNode node : graph.GetNodes()) {
    has_gather = has_gather || IsOnnxOp(node, "Gather");
    has_greater = has_greater || IsOnnxOp(node, "Greater");
    has_squeeze = has_squeeze || IsOnnxOp(node, "Squeeze");
    has_sub = has_sub || IsOnnxOp(node, "Sub");
  }
  return has_gather && has_greater && has_squeeze && has_sub;
}

std::unique_ptr<FusionNodeCompute> CreateBucketizeGatherFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  auto producers = ProducersInGraph(graph);
  Ort::ConstNode squeeze_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Squeeze")) {
      squeeze_node = node;
      break;
    }
  }
  if (!squeeze_node) {
    throw std::runtime_error("BucketizeGather requires Squeeze");
  }

  std::vector<Ort::ConstValueInfo> squeeze_inputs = squeeze_node.GetInputs();
  std::vector<Ort::ConstValueInfo> squeeze_outputs = squeeze_node.GetOutputs();
  if (squeeze_inputs.size() != 2 || squeeze_outputs.size() != 1) {
    throw std::runtime_error("invalid BucketizeGather Squeeze");
  }
  auto squeeze_axes = ReadSmallIntInitializer(squeeze_inputs[1]);
  if (!squeeze_axes.has_value() || squeeze_axes->size() != 1) {
    throw std::runtime_error("BucketizeGather requires one axis");
  }
  int64_t squeeze_axis = (*squeeze_axes)[0];

  auto gather_it = producers.find(Name(squeeze_inputs[0]));
  if (gather_it == producers.end() || !IsOnnxOp(gather_it->second, "Gather")) {
    throw std::runtime_error("BucketizeGather requires Gather");
  }
  Ort::ConstNode gather_node = gather_it->second;
  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  if (gather_inputs.size() != 2) {
    throw std::runtime_error("invalid BucketizeGather Gather");
  }

  auto final_mul_it = producers.find(Name(gather_inputs[1]));
  if (final_mul_it == producers.end() ||
      !IsOnnxOp(final_mul_it->second, "Mul")) {
    throw std::runtime_error("BucketizeGather requires final Mul indices");
  }
  Ort::ConstNode final_mul = final_mul_it->second;
  std::vector<Ort::ConstValueInfo> final_mul_inputs = final_mul.GetInputs();
  if (final_mul_inputs.size() != 2) {
    throw std::runtime_error("invalid BucketizeGather final Mul");
  }

  Ort::ConstNode add_node{nullptr};
  Ort::ConstNode mask_cast_node{nullptr};
  for (Ort::ConstValueInfo input : final_mul_inputs) {
    auto it = producers.find(Name(input));
    if (it != producers.end() && IsOnnxOp(it->second, "Add")) {
      add_node = it->second;
    } else if (it != producers.end() && IsOnnxOp(it->second, "Cast")) {
      mask_cast_node = it->second;
    }
  }
  if (!add_node || !mask_cast_node) {
    throw std::runtime_error(
        "BucketizeGather final Mul must combine Add and Cast");
  }

  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  if (add_inputs.size() != 2) {
    throw std::runtime_error("invalid BucketizeGather Add");
  }
  Ort::ConstNode sub_node{nullptr};
  Ort::ConstValueInfo offset_input{nullptr};
  for (Ort::ConstValueInfo input : add_inputs) {
    auto it = producers.find(Name(input));
    if (it != producers.end() && IsOnnxOp(it->second, "Sub")) {
      sub_node = it->second;
    } else {
      offset_input = input;
    }
  }
  if (!sub_node || offset_input == nullptr) {
    throw std::runtime_error("BucketizeGather Add must combine Sub and offset");
  }

  std::vector<Ort::ConstValueInfo> sub_inputs = sub_node.GetInputs();
  if (sub_inputs.size() != 2) {
    throw std::runtime_error("invalid BucketizeGather Sub");
  }
  Ort::ConstValueInfo source_input = sub_inputs[0];
  auto product_it = producers.find(Name(sub_inputs[1]));
  if (product_it == producers.end() || !IsOnnxOp(product_it->second, "Mul")) {
    throw std::runtime_error("BucketizeGather Sub must subtract Div*modulus");
  }
  Ort::ConstNode product_mul = product_it->second;
  std::vector<Ort::ConstValueInfo> product_inputs = product_mul.GetInputs();
  if (product_inputs.size() != 2) {
    throw std::runtime_error("invalid BucketizeGather product Mul");
  }

  Ort::ConstNode div_node{nullptr};
  Ort::ConstValueInfo modulus_input{nullptr};
  for (Ort::ConstValueInfo input : product_inputs) {
    auto it = producers.find(Name(input));
    if (it != producers.end() && IsOnnxOp(it->second, "Div")) {
      div_node = it->second;
    } else {
      modulus_input = input;
    }
  }
  if (!div_node || modulus_input == nullptr) {
    throw std::runtime_error(
        "BucketizeGather product Mul must combine Div and modulus");
  }

  std::vector<Ort::ConstValueInfo> mask_cast_inputs =
      mask_cast_node.GetInputs();
  if (mask_cast_inputs.size() != 1) {
    throw std::runtime_error("invalid BucketizeGather mask Cast");
  }
  auto greater_it = producers.find(Name(mask_cast_inputs[0]));
  if (greater_it == producers.end() ||
      !IsOnnxOp(greater_it->second, "Greater")) {
    throw std::runtime_error("BucketizeGather mask Cast must consume Greater");
  }
  Ort::ConstNode greater_node = greater_it->second;
  std::vector<Ort::ConstValueInfo> greater_inputs = greater_node.GetInputs();
  if (greater_inputs.size() != 2) {
    throw std::runtime_error("invalid BucketizeGather Greater");
  }

  Ort::ConstValueInfo threshold_input{nullptr};
  Ort::ConstNode source_cast_node{nullptr};
  for (Ort::ConstValueInfo input : greater_inputs) {
    auto it = producers.find(Name(input));
    if (it != producers.end() && IsOnnxOp(it->second, "Cast")) {
      source_cast_node = it->second;
    } else {
      threshold_input = input;
    }
  }
  if (!source_cast_node || threshold_input == nullptr) {
    throw std::runtime_error(
        "BucketizeGather Greater must compare Cast and threshold");
  }
  std::vector<Ort::ConstValueInfo> source_cast_inputs =
      source_cast_node.GetInputs();
  if (source_cast_inputs.size() != 1 ||
      Name(source_cast_inputs[0]) != Name(source_input)) {
    throw std::runtime_error(
        "BucketizeGather Greater Cast must consume source input");
  }

  auto fused_inputs = FusedInputIndices(fused_node);
  auto fused_outputs = FusedOutputIndices(fused_node);
  return std::make_unique<BucketizeGatherFusionCompute>(
      GetMappedIndex(fused_inputs, Name(gather_inputs[0]), "table input"),
      GetMappedIndex(fused_inputs, Name(source_input), "source input"),
      ReadRequiredScalarIntInitializer(modulus_input, "modulus"),
      ReadRequiredScalarIntInitializer(offset_input, "offset"),
      ReadRequiredScalarFloatInitializer(threshold_input, "threshold"),
      squeeze_axis,
      GetMappedIndex(fused_outputs, Name(squeeze_outputs[0]), "output"));
}
