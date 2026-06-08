// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/tile_mask_select_fusion.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/where_impl.h"

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

Ort::ConstNode ProducerOf(Ort::ConstValueInfo value_info) {
  Ort::ValueInfoConsumerProducerInfo producer = value_info.GetProducerNode();
  return producer.node;
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
    throw std::runtime_error(std::string("unable to map TileMaskSelect ") +
                             kind + " " + name);
  }
  return it->second;
}

Ort::ConstNode FindTileNode(Ort::ConstGraph graph) {
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Tile")) {
      return node;
    }
  }
  return Ort::ConstNode{nullptr};
}

bool IsFloatCast(Ort::ConstNode node) {
  return node && IsOnnxOp(node, "Cast") &&
         ReadIntAttribute(node, "to", 0) == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

bool FindMaskCastsFromValue(Ort::ConstValueInfo mask_value,
                            Ort::ConstNode& mask_cast_node,
                            Ort::ConstNode& inverse_cast_node) {
  Ort::ConstNode not_node{nullptr};
  for (const auto& consumer : mask_value.GetConsumers()) {
    if (consumer.index != 0) {
      continue;
    }
    if (IsFloatCast(consumer.node)) {
      mask_cast_node = consumer.node;
    } else if (IsOnnxOp(consumer.node, "Not")) {
      not_node = consumer.node;
    } else {
      return false;
    }
  }
  if (!mask_cast_node || !not_node) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> not_outputs = not_node.GetOutputs();
  if (not_outputs.size() != 1) {
    return false;
  }
  std::vector<Ort::ValueInfoConsumerProducerInfo> not_consumers =
      not_outputs[0].GetConsumers();
  if (not_consumers.size() != 1 || not_consumers[0].index != 0 ||
      !IsFloatCast(not_consumers[0].node)) {
    return false;
  }
  inverse_cast_node = not_consumers[0].node;
  return true;
}

bool FindMaskCasts(Ort::ConstNode tile_node, Ort::ConstNode& mask_cast_node,
                   Ort::ConstNode& inverse_cast_node,
                   Ort::ConstValueInfo& mask_input) {
  std::vector<Ort::ConstValueInfo> tile_inputs = tile_node.GetInputs();
  std::vector<Ort::ConstValueInfo> tile_outputs = tile_node.GetOutputs();
  if (tile_inputs.size() != 2 || tile_outputs.size() != 1) {
    return false;
  }
  mask_input = tile_inputs[0];
  return FindMaskCastsFromValue(tile_outputs[0], mask_cast_node,
                                inverse_cast_node);
}

bool FindDirectMaskCasts(Ort::ConstGraph graph, Ort::ConstNode& mask_cast_node,
                         Ort::ConstNode& inverse_cast_node,
                         Ort::ConstValueInfo& mask_input) {
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsFloatCast(node)) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    if (inputs.size() != 1) {
      continue;
    }
    Ort::ConstNode found_mask_cast{nullptr};
    Ort::ConstNode found_inverse_cast{nullptr};
    if (!FindMaskCastsFromValue(inputs[0], found_mask_cast,
                                found_inverse_cast)) {
      continue;
    }
    mask_cast_node = found_mask_cast;
    inverse_cast_node = found_inverse_cast;
    mask_input = inputs[0];
    return true;
  }
  return false;
}

bool FindMaskSelectCasts(Ort::ConstGraph graph, Ort::ConstNode& mask_cast_node,
                         Ort::ConstNode& inverse_cast_node,
                         Ort::ConstValueInfo& mask_input) {
  Ort::ConstNode tile_node = FindTileNode(graph);
  if (tile_node &&
      FindMaskCasts(tile_node, mask_cast_node, inverse_cast_node,
                    mask_input)) {
    return true;
  }
  return FindDirectMaskCasts(graph, mask_cast_node, inverse_cast_node,
                             mask_input);
}

bool MulUsesValue(Ort::ConstNode mul_node, const std::string& value_name,
                  Ort::ConstValueInfo& other_input) {
  if (!IsOnnxOp(mul_node, "Mul")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> inputs = mul_node.GetInputs();
  if (inputs.size() != 2) {
    return false;
  }
  if (Name(inputs[0]) == value_name) {
    other_input = inputs[1];
    return true;
  }
  if (Name(inputs[1]) == value_name) {
    other_input = inputs[0];
    return true;
  }
  return false;
}

Ort::ConstNode SingleAddConsumer(Ort::ConstNode mul_node) {
  std::vector<Ort::ConstValueInfo> outputs = mul_node.GetOutputs();
  if (outputs.size() != 1) {
    return Ort::ConstNode{nullptr};
  }
  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
      outputs[0].GetConsumers();
  if (consumers.size() != 1 || !IsOnnxOp(consumers[0].node, "Add")) {
    return Ort::ConstNode{nullptr};
  }
  return consumers[0].node;
}

std::vector<TileMaskSelectPlan> BuildOutputPlans(Ort::ConstGraph graph,
                                                 Ort::ConstNode fused_node,
                                                 Ort::ConstNode mask_cast_node,
                                                 Ort::ConstNode inverse_cast_node) {
  auto fused_input_indices = FusedInputIndices(fused_node);
  auto fused_output_indices = FusedOutputIndices(fused_node);

  const std::string mask_value = Name(mask_cast_node.GetOutputs()[0]);
  const std::string inverse_value = Name(inverse_cast_node.GetOutputs()[0]);
  std::vector<TileMaskSelectPlan> outputs;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, "Add")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> add_inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> add_outputs = node.GetOutputs();
    if (add_inputs.size() != 2 || add_outputs.size() != 1) {
      continue;
    }

    Ort::ConstNode lhs_mul = ProducerOf(add_inputs[0]);
    Ort::ConstNode rhs_mul = ProducerOf(add_inputs[1]);
    Ort::ConstValueInfo true_input{nullptr};
    Ort::ConstValueInfo false_input{nullptr};
    const bool lhs_mask = MulUsesValue(lhs_mul, mask_value, true_input);
    const bool rhs_inverse = MulUsesValue(rhs_mul, inverse_value, false_input);
    const bool lhs_inverse = MulUsesValue(lhs_mul, inverse_value, false_input);
    const bool rhs_mask = MulUsesValue(rhs_mul, mask_value, true_input);
    if (!((lhs_mask && rhs_inverse) || (lhs_inverse && rhs_mask))) {
      continue;
    }

    outputs.push_back(TileMaskSelectPlan{
        GetMappedIndex(fused_input_indices, Name(true_input), "input"),
        GetMappedIndex(fused_input_indices, Name(false_input), "input"),
        GetMappedIndex(fused_output_indices, Name(add_outputs[0]), "output")});
  }
  return outputs;
}

void FillBroadcastStrides(const std::vector<int64_t>& out_shape,
                          const std::vector<int64_t>& input_shape,
                          int64_t* dst) {
  std::fill(dst, dst + kMusaMaxBroadcastRank, 0);
  const size_t rank = out_shape.size();
  const size_t input_rank = input_shape.size();
  const size_t offset = rank - input_rank;
  auto input_strides = Strides(input_shape);
  for (size_t dim = 0; dim < rank; ++dim) {
    if (dim < offset) {
      dst[dim] = 0;
      continue;
    }
    const size_t input_dim = dim - offset;
    dst[dim] = input_shape[input_dim] == 1 ? 0 : input_strides[input_dim];
  }
}

MusaWhereParams MakeWhereParams(const std::vector<int64_t>& out_shape,
                                const std::vector<int64_t>& condition_shape,
                                const std::vector<int64_t>& x_shape,
                                const std::vector<int64_t>& y_shape) {
  MusaWhereParams params{};
  params.rank = static_cast<int32_t>(out_shape.size());
  params.total_elements = NumElements(out_shape);
  auto output_strides = Strides(out_shape);
  for (size_t dim = 0; dim < out_shape.size(); ++dim) {
    params.output_strides[dim] = output_strides[dim];
  }
  FillBroadcastStrides(out_shape, condition_shape, params.condition_strides);
  FillBroadcastStrides(out_shape, x_shape, params.x_strides);
  FillBroadcastStrides(out_shape, y_shape, params.y_strides);
  return params;
}

bool SameShape(const std::vector<int64_t>& lhs,
               const std::vector<int64_t>& rhs) {
  return lhs == rhs;
}

bool FastWhereElementSize(size_t element_size) {
  return element_size == 1 || element_size == 2 || element_size == 4 ||
         element_size == 8;
}

bool RowwiseDataShape(const std::vector<int64_t>& input_shape,
                      const std::vector<int64_t>& out_shape,
                      bool& broadcast_rows) {
  if (input_shape.size() != out_shape.size() || input_shape.empty()) {
    return false;
  }
  for (size_t dim = 1; dim < out_shape.size(); ++dim) {
    if (input_shape[dim] != out_shape[dim]) {
      return false;
    }
  }
  if (input_shape[0] == out_shape[0]) {
    broadcast_rows = false;
    return true;
  }
  if (input_shape[0] == 1) {
    broadcast_rows = true;
    return true;
  }
  return false;
}

OrtStatus* TryLaunchFastWhere(const uint8_t* mask_data,
                              const void* true_data,
                              const void* false_data,
                              void* output_data,
                              size_t elem_size,
                              const std::vector<int64_t>& mask_shape,
                              const std::vector<int64_t>& true_shape,
                              const std::vector<int64_t>& false_shape,
                              const std::vector<int64_t>& out_shape,
                              bool& launched) {
  launched = false;
  if (!FastWhereElementSize(elem_size)) {
    return nullptr;
  }

  if (SameShape(mask_shape, out_shape) && SameShape(true_shape, out_shape) &&
      SameShape(false_shape, out_shape)) {
    launched = true;
    return LaunchStatus(LaunchMusaWhereSameShapeFastKernel(
        mask_data, true_data, false_data, output_data,
        static_cast<int32_t>(elem_size), NumElements(out_shape), nullptr));
  }

  if (out_shape.size() < 2 || mask_shape.size() != 2 ||
      mask_shape[0] != out_shape[0] || mask_shape[1] != 1) {
    return nullptr;
  }

  bool true_broadcast_rows = false;
  bool false_broadcast_rows = false;
  if (!RowwiseDataShape(true_shape, out_shape, true_broadcast_rows) ||
      !RowwiseDataShape(false_shape, out_shape, false_broadcast_rows)) {
    return nullptr;
  }

  const int64_t rows = out_shape[0];
  const int64_t inner_size = NumElements(out_shape) / rows;
  launched = true;
  return LaunchStatus(LaunchMusaWhereRowwiseFastKernel(
      mask_data, true_data, false_data, output_data,
      static_cast<int32_t>(elem_size), rows, inner_size, true_broadcast_rows,
      false_broadcast_rows, nullptr));
}

}  // namespace

TileMaskSelectFusionCompute::TileMaskSelectFusionCompute(
    size_t mask_input_index, std::vector<TileMaskSelectPlan> outputs)
    : mask_input_index(mask_input_index), outputs(std::move(outputs)) {}

OrtStatus* TileMaskSelectFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    Ort::ConstValue mask = ctx.GetInput(mask_input_index);
    auto mask_info = mask.GetTensorTypeAndShapeInfo();
    if (mask_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL ||
        !IsGpuMemory(mask.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "TileMaskSelect requires MUSA bool mask");
    }
    std::vector<int64_t> mask_shape = mask_info.GetShape();
    if (mask_shape.empty() || mask_shape.size() > kMusaMaxBroadcastRank ||
        mask_shape.back() != 1) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "TileMaskSelect requires mask last dimension to be 1");
    }

    for (const TileMaskSelectPlan& plan : outputs) {
      Ort::ConstValue true_value = ctx.GetInput(plan.true_input_index);
      Ort::ConstValue false_value = ctx.GetInput(plan.false_input_index);
      auto true_info = true_value.GetTensorTypeAndShapeInfo();
      auto false_info = false_value.GetTensorTypeAndShapeInfo();
      if (true_info.GetElementType() != false_info.GetElementType() ||
          !IsGpuMemory(true_value.GetTensorMemoryInfo()) ||
          !IsGpuMemory(false_value.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "TileMaskSelect requires matching MUSA data inputs");
      }
      std::vector<int64_t> true_shape = true_info.GetShape();
      std::vector<int64_t> false_shape = false_info.GetShape();
      std::vector<int64_t> out_shape =
          BroadcastShape(BroadcastShape(mask_shape, true_shape), false_shape);
      if (out_shape.size() > kMusaMaxBroadcastRank ||
          mask_shape.size() > kMusaMaxBroadcastRank ||
          true_shape.size() > kMusaMaxBroadcastRank ||
          false_shape.size() > kMusaMaxBroadcastRank) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "TileMaskSelect rank exceeds MUSA limit");
      }

      const size_t elem_size = ElementSize(true_info.GetElementType());
      if (elem_size == 0) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "TileMaskSelect unsupported dtype");
      }
      Ort::UnownedValue output = ctx.GetOutput(plan.output_index, out_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "TileMaskSelect requires MUSA output");
      }

      bool launched_fast = false;
      OrtStatus* fast_status = TryLaunchFastWhere(
          mask.GetTensorData<uint8_t>(), true_value.GetTensorRawData(),
          false_value.GetTensorRawData(), output.GetTensorMutableRawData(),
          elem_size, mask_shape, true_shape, false_shape, out_shape,
          launched_fast);
      if (fast_status != nullptr) {
        return fast_status;
      }
      if (launched_fast) {
        continue;
      }

      OrtStatus* status = LaunchStatus(LaunchMusaWhereKernel(
          mask.GetTensorData<uint8_t>(), true_value.GetTensorRawData(),
          false_value.GetTensorRawData(), output.GetTensorMutableRawData(),
          static_cast<int32_t>(elem_size),
          MakeWhereParams(out_shape, mask_shape, true_shape, false_shape),
          nullptr));
      if (status != nullptr) {
        return status;
      }
    }
    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

bool IsTileMaskSelectFusionGraph(Ort::ConstGraph graph) {
  Ort::ConstNode mask_cast_node{nullptr};
  Ort::ConstNode inverse_cast_node{nullptr};
  Ort::ConstValueInfo mask_input{nullptr};
  if (!FindMaskSelectCasts(graph, mask_cast_node, inverse_cast_node,
                           mask_input)) {
    return false;
  }
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Add")) {
      return true;
    }
  }
  return false;
}

std::unique_ptr<FusionNodeCompute> CreateTileMaskSelectFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode mask_cast_node{nullptr};
  Ort::ConstNode inverse_cast_node{nullptr};
  Ort::ConstValueInfo mask_input{nullptr};
  if (!FindMaskSelectCasts(graph, mask_cast_node, inverse_cast_node,
                           mask_input)) {
    throw std::runtime_error(
        "TileMaskSelect fusion expects Cast/Not/Cast mask select");
  }
  std::vector<TileMaskSelectPlan> outputs =
      BuildOutputPlans(graph, fused_node, mask_cast_node, inverse_cast_node);
  if (outputs.empty()) {
    throw std::runtime_error("TileMaskSelect fusion has no select outputs");
  }
  auto fused_input_indices = FusedInputIndices(fused_node);
  return std::make_unique<TileMaskSelectFusionCompute>(
      GetMappedIndex(fused_input_indices, Name(mask_input), "input"),
      std::move(outputs));
}
