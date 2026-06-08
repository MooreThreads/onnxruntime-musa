// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/masked_gather_reduce_fusion.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/reduction/masked_gather_reduce_impl.h"
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
        "MaskedGatherReduce requires constant axes input");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("MaskedGatherReduce failed to read initializer");
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
  throw std::runtime_error("MaskedGatherReduce axes must be int32/int64");
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
    throw std::runtime_error(std::string("unable to map MaskedGatherReduce ") +
                             kind + " " + name);
  }
  return it->second;
}

bool HasSingleConsumer(Ort::ConstValueInfo value_info, const char* op_type,
                       Ort::ConstNode& consumer_node, int expected_input) {
  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
      value_info.GetConsumers();
  if (consumers.size() != 1 ||
      consumers[0].index != expected_input ||
      !IsOnnxOp(consumers[0].node, op_type)) {
    return false;
  }
  consumer_node = consumers[0].node;
  return true;
}

bool NodeHasOneInputOutput(Ort::ConstNode node) {
  return node.GetInputs().size() >= 1 && node.GetOutputs().size() == 1;
}

bool AxesInitializerIs(Ort::ConstValueInfo value_info, int64_t axis) {
  try {
    std::vector<int64_t> axes = ReadIntInitializer(value_info);
    return axes.size() == 1 && axes[0] == axis;
  } catch (const std::exception&) {
    return false;
  }
}

bool FindMaskedGatherReduceNodes(Ort::ConstGraph graph,
                                 Ort::ConstNode& nonzero_node,
                                 Ort::ConstNode& transpose_node,
                                 Ort::ConstNode& squeeze_node,
                                 Ort::ConstNode& gather_node,
                                 Ort::ConstNode& reduce_node) {
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, "NonZero")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> nonzero_outputs = node.GetOutputs();
    if (nonzero_outputs.size() != 1 ||
        !HasSingleConsumer(nonzero_outputs[0], "Transpose", transpose_node, 0)) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> transpose_outputs =
        transpose_node.GetOutputs();
    if (!NodeHasOneInputOutput(transpose_node) ||
        !HasSingleConsumer(transpose_outputs[0], "Squeeze", squeeze_node, 0)) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> squeeze_inputs = squeeze_node.GetInputs();
    std::vector<Ort::ConstValueInfo> squeeze_outputs =
        squeeze_node.GetOutputs();
    if (squeeze_inputs.size() != 2 || squeeze_outputs.size() != 1 ||
        !AxesInitializerIs(squeeze_inputs[1], 1) ||
        !HasSingleConsumer(squeeze_outputs[0], "Gather", gather_node, 1)) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> gather_outputs = gather_node.GetOutputs();
    if (gather_node.GetInputs().size() != 2 || gather_outputs.size() != 1 ||
        ReadIntAttribute(gather_node, "axis", 0) != 0) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> reduce_consumers =
        gather_outputs[0].GetConsumers();
    if (reduce_consumers.size() != 1 || reduce_consumers[0].index != 0) {
      continue;
    }
    Ort::ConstNode candidate_reduce = reduce_consumers[0].node;
    if (!IsOnnxOp(candidate_reduce, "ReduceProd") &&
        !IsOnnxOp(candidate_reduce, "ReduceMean")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> reduce_inputs =
        candidate_reduce.GetInputs();
    if (reduce_inputs.size() != 2 ||
        !AxesInitializerIs(reduce_inputs[1], 0) ||
        ReadIntAttribute(candidate_reduce, "keepdims", 1) != 0) {
      continue;
    }

    nonzero_node = node;
    reduce_node = candidate_reduce;
    return true;
  }
  return false;
}

MusaReduceOp ReduceOpForNode(Ort::ConstNode node) {
  if (IsOnnxOp(node, "ReduceMean")) {
    return MusaReduceOp::Mean;
  }
  if (IsOnnxOp(node, "ReduceProd")) {
    return MusaReduceOp::Prod;
  }
  throw std::runtime_error("MaskedGatherReduce unsupported reduction op");
}

}  // namespace

MaskedGatherReduceFusionCompute::MaskedGatherReduceFusionCompute(
    size_t mask_input_index, size_t data_input_index, size_t output_index,
    MusaReduceOp reduce_op)
    : mask_input_index(mask_input_index),
      data_input_index(data_input_index),
      output_index(output_index),
      reduce_op(reduce_op) {}

OrtStatus* MaskedGatherReduceFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    Ort::ConstValue mask = ctx.GetInput(mask_input_index);
    Ort::ConstValue data = ctx.GetInput(data_input_index);
    auto mask_info = mask.GetTensorTypeAndShapeInfo();
    auto data_info = data.GetTensorTypeAndShapeInfo();
    if (mask_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL ||
        data_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "MaskedGatherReduce requires bool mask and float data");
    }
    if (!IsGpuMemory(mask.GetTensorMemoryInfo()) ||
        !IsGpuMemory(data.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "MaskedGatherReduce requires MUSA device inputs");
    }
    std::vector<int64_t> mask_shape = mask_info.GetShape();
    std::vector<int64_t> data_shape = data_info.GetShape();
    if (mask_shape.size() != 1 || data_shape.size() != 1 ||
        mask_shape[0] != data_shape[0]) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "MaskedGatherReduce requires matching 1D mask/data tensors");
    }

    std::vector<int64_t> output_shape;
    Ort::UnownedValue output = ctx.GetOutput(output_index, output_shape);
    if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "MaskedGatherReduce requires MUSA output");
    }
    return LaunchStatus(LaunchMusaMaskedGatherReduceFloatKernel(
        mask.GetTensorData<uint8_t>(), data.GetTensorData<float>(),
        output.GetTensorMutableData<float>(), mask_shape[0], reduce_op,
        nullptr));
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

bool IsMaskedGatherReduceFusionGraph(Ort::ConstGraph graph) {
  Ort::ConstNode nonzero_node{nullptr};
  Ort::ConstNode transpose_node{nullptr};
  Ort::ConstNode squeeze_node{nullptr};
  Ort::ConstNode gather_node{nullptr};
  Ort::ConstNode reduce_node{nullptr};
  return FindMaskedGatherReduceNodes(graph, nonzero_node, transpose_node,
                                     squeeze_node, gather_node, reduce_node);
}

std::unique_ptr<FusionNodeCompute> CreateMaskedGatherReduceFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode nonzero_node{nullptr};
  Ort::ConstNode transpose_node{nullptr};
  Ort::ConstNode squeeze_node{nullptr};
  Ort::ConstNode gather_node{nullptr};
  Ort::ConstNode reduce_node{nullptr};
  if (!FindMaskedGatherReduceNodes(graph, nonzero_node, transpose_node,
                                   squeeze_node, gather_node, reduce_node)) {
    throw std::runtime_error(
        "MaskedGatherReduce fusion expects NonZero+Transpose+Squeeze+Gather+Reduce");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  auto fused_output_indices = FusedOutputIndices(fused_node);
  std::vector<Ort::ConstValueInfo> nonzero_inputs = nonzero_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reduce_outputs = reduce_node.GetOutputs();
  if (nonzero_inputs.size() != 1 || gather_inputs.size() != 2 ||
      reduce_outputs.size() != 1) {
    throw std::runtime_error("MaskedGatherReduce node input/output mismatch");
  }

  return std::make_unique<MaskedGatherReduceFusionCompute>(
      GetMappedIndex(fused_input_indices, Name(nonzero_inputs[0]), "input"),
      GetMappedIndex(fused_input_indices, Name(gather_inputs[0]), "input"),
      GetMappedIndex(fused_output_indices, Name(reduce_outputs[0]), "output"),
      ReduceOpForNode(reduce_node));
}
