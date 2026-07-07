// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/modulo_gather_fusion.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/modulo_gather_impl.h"

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
    throw std::runtime_error(std::string("unable to map ModuloGather ") + kind +
                             " " + name);
  }
  return it->second;
}

int64_t ReadScalarInt(Ort::ConstValue value, musaStream_t stream,
                      const char* name) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementCount() != 1) {
    throw std::runtime_error(std::string("ModuloGather ") + name +
                             " must be scalar or single-element tensor");
  }
  std::vector<uint8_t> bytes;
  Ort::ThrowOnError(CopyToHost(value, bytes, stream));
  ONNXTensorElementDataType elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    int64_t result = 0;
    std::memcpy(&result, bytes.data(), sizeof(result));
    return result;
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    int32_t result = 0;
    std::memcpy(&result, bytes.data(), sizeof(result));
    return static_cast<int64_t>(result);
  }
  throw std::runtime_error(std::string("ModuloGather ") + name +
                           " must be int32/int64");
}

struct ModuloGatherFusionCompute : FusionNodeCompute {
  ModuloGatherFusionCompute(size_t table_index, size_t input_index,
                            size_t modulus_index, size_t offset_index,
                            size_t invalid_index, size_t output_index)
      : table_index(table_index),
        input_index(input_index),
        modulus_index(modulus_index),
        offset_index(offset_index),
        invalid_index(invalid_index),
        output_index(output_index) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue table = ctx.GetInput(table_index);
      Ort::ConstValue input = ctx.GetInput(input_index);
      Ort::ConstValue modulus_value = ctx.GetInput(modulus_index);
      Ort::ConstValue offset_value = ctx.GetInput(offset_index);
      Ort::ConstValue invalid_value = ctx.GetInput(invalid_index);

      if (!IsGpuMemory(table.GetTensorMemoryInfo()) ||
          !IsGpuMemory(input.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "ModuloGather requires MUSA table and index tensors");
      }

      auto table_info = table.GetTensorTypeAndShapeInfo();
      auto input_info = input.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> table_shape = table_info.GetShape();
      std::vector<int64_t> input_shape = input_info.GetShape();
      if (table_shape.empty() || table_shape[0] <= 0) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "ModuloGather table must have rows");
      }

      const size_t elem_size = ElementSize(table_info.GetElementType());
      if (elem_size == 0) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "ModuloGather unsupported table dtype");
      }
      const size_t index_elem_size = ElementSize(input_info.GetElementType());
      if (index_elem_size != sizeof(int32_t) &&
          index_elem_size != sizeof(int64_t)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "ModuloGather unsupported index dtype");
      }

      std::vector<int64_t> out_shape = input_shape;
      out_shape.insert(out_shape.end(), table_shape.begin() + 1,
                       table_shape.end());
      Ort::UnownedValue output = ctx.GetOutput(output_index, out_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                          "ModuloGather requires MUSA output");
      }

      int64_t block_size = 1;
      for (size_t i = 1; i < table_shape.size(); ++i) {
        block_size *= table_shape[i];
      }
      const int64_t indices_count = NumElements(input_shape);
      const int64_t modulus =
          ReadScalarInt(modulus_value, GetComputeStream(ctx), "modulus");
      const int64_t offset =
          ReadScalarInt(offset_value, GetComputeStream(ctx), "offset");
      const int64_t invalid =
          ReadScalarInt(invalid_value, GetComputeStream(ctx), "invalid value");
      if (modulus <= 0 || offset < 0 || offset + modulus > table_shape[0]) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "ModuloGather invalid modulus/offset");
      }

      DeviceInputBuffer table_buffer;
      DeviceInputBuffer input_buffer;
      RETURN_IF_ERROR(table_buffer.Bind(table, GetComputeStream(ctx)));
      RETURN_IF_ERROR(input_buffer.Bind(input, GetComputeStream(ctx)));
      return LaunchStatus(LaunchMusaModuloGatherKernel(
          table_buffer.data(), input_buffer.data(),
          output.GetTensorMutableRawData(), static_cast<int32_t>(elem_size),
          static_cast<int32_t>(index_elem_size), indices_count, table_shape[0],
          block_size, modulus, offset, invalid, GetComputeStream(ctx)));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t table_index;
  size_t input_index;
  size_t modulus_index;
  size_t offset_index;
  size_t invalid_index;
  size_t output_index;
};

}  // namespace

bool IsModuloGatherFusionGraph(Ort::ConstGraph graph) {
  bool has_gather = false;
  bool has_equal = false;
  bool has_sub = false;
  for (Ort::ConstNode node : graph.GetNodes()) {
    has_gather = has_gather || IsOnnxOp(node, "Gather");
    has_equal = has_equal || IsOnnxOp(node, "Equal");
    has_sub = has_sub || IsOnnxOp(node, "Sub");
  }
  return has_gather && has_equal && has_sub;
}

std::unique_ptr<FusionNodeCompute> CreateModuloGatherFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  auto producers = ProducersInGraph(graph);
  Ort::ConstNode gather_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Gather")) {
      gather_node = node;
      break;
    }
  }
  if (!gather_node) {
    throw std::runtime_error("ModuloGather requires Gather");
  }

  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gather_outputs = gather_node.GetOutputs();
  if (gather_inputs.size() != 2 || gather_outputs.size() != 1) {
    throw std::runtime_error("invalid ModuloGather Gather node");
  }

  auto final_mul_it = producers.find(Name(gather_inputs[1]));
  if (final_mul_it == producers.end() ||
      !IsOnnxOp(final_mul_it->second, "Mul")) {
    throw std::runtime_error("ModuloGather requires final Mul indices");
  }
  Ort::ConstNode final_mul = final_mul_it->second;
  std::vector<Ort::ConstValueInfo> final_mul_inputs = final_mul.GetInputs();
  if (final_mul_inputs.size() != 2) {
    throw std::runtime_error("invalid ModuloGather final Mul");
  }

  Ort::ConstNode add_node{nullptr};
  Ort::ConstNode cast_node{nullptr};
  for (Ort::ConstValueInfo input : final_mul_inputs) {
    auto it = producers.find(Name(input));
    if (it != producers.end() && IsOnnxOp(it->second, "Add")) {
      add_node = it->second;
    } else if (it != producers.end() && IsOnnxOp(it->second, "Cast")) {
      cast_node = it->second;
    }
  }
  if (!add_node || !cast_node) {
    throw std::runtime_error(
        "ModuloGather final Mul must combine Add and Cast");
  }

  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  if (add_inputs.size() != 2) {
    throw std::runtime_error("invalid ModuloGather Add");
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
    throw std::runtime_error("ModuloGather Add must combine Sub and offset");
  }

  std::vector<Ort::ConstValueInfo> sub_inputs = sub_node.GetInputs();
  if (sub_inputs.size() != 2) {
    throw std::runtime_error("invalid ModuloGather Sub");
  }
  Ort::ConstValueInfo source_input = sub_inputs[0];
  auto product_it = producers.find(Name(sub_inputs[1]));
  if (product_it == producers.end() || !IsOnnxOp(product_it->second, "Mul")) {
    throw std::runtime_error("ModuloGather Sub must subtract Div*modulus");
  }
  Ort::ConstNode product_mul = product_it->second;
  std::vector<Ort::ConstValueInfo> product_inputs = product_mul.GetInputs();
  if (product_inputs.size() != 2) {
    throw std::runtime_error("invalid ModuloGather product Mul");
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
        "ModuloGather product Mul must combine Div and modulus");
  }

  std::vector<Ort::ConstValueInfo> cast_inputs = cast_node.GetInputs();
  if (cast_inputs.size() != 1) {
    throw std::runtime_error("invalid ModuloGather Cast");
  }
  auto not_it = producers.find(Name(cast_inputs[0]));
  if (not_it == producers.end() || !IsOnnxOp(not_it->second, "Not")) {
    throw std::runtime_error("ModuloGather Cast must consume Not");
  }
  Ort::ConstNode not_node = not_it->second;
  std::vector<Ort::ConstValueInfo> not_inputs = not_node.GetInputs();
  if (not_inputs.size() != 1) {
    throw std::runtime_error("invalid ModuloGather Not");
  }
  auto equal_it = producers.find(Name(not_inputs[0]));
  if (equal_it == producers.end() || !IsOnnxOp(equal_it->second, "Equal")) {
    throw std::runtime_error("ModuloGather Not must consume Equal");
  }
  Ort::ConstNode equal_node = equal_it->second;
  std::vector<Ort::ConstValueInfo> equal_inputs = equal_node.GetInputs();
  if (equal_inputs.size() != 2) {
    throw std::runtime_error("invalid ModuloGather Equal");
  }
  Ort::ConstValueInfo invalid_input =
      Name(equal_inputs[0]) == Name(source_input) ? equal_inputs[1]
                                                  : equal_inputs[0];

  auto fused_inputs = FusedInputIndices(fused_node);
  auto fused_outputs = FusedOutputIndices(fused_node);
  return std::make_unique<ModuloGatherFusionCompute>(
      GetMappedIndex(fused_inputs, Name(gather_inputs[0]), "table input"),
      GetMappedIndex(fused_inputs, Name(source_input), "source input"),
      GetMappedIndex(fused_inputs, Name(modulus_input), "modulus input"),
      GetMappedIndex(fused_inputs, Name(offset_input), "offset input"),
      GetMappedIndex(fused_inputs, Name(invalid_input), "invalid input"),
      GetMappedIndex(fused_outputs, Name(gather_outputs[0]), "output"));
}
