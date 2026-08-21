// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "fusion/generate_mtgr_custom_mask_fusion.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/generate_mtgr_custom_mask_impl.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

std::unordered_map<std::string, size_t> ValueIndices(
    const std::vector<Ort::ConstValueInfo>& values) {
  std::unordered_map<std::string, size_t> indices;
  for (size_t i = 0; i < values.size(); ++i) {
    indices.emplace(Name(values[i]), i);
  }
  return indices;
}

size_t RequiredIndex(const std::unordered_map<std::string, size_t>& indices,
                     const std::string& name, const char* kind) {
  auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error(std::string("Generate MTGR custom mask missing ") +
                             kind + " " + name);
  }
  return it->second;
}

bool IsScalarInt64(Ort::ConstValueInfo value) {
  auto type = GetTensorElementType(value);
  auto shape = GetTensorShape(value);
  return type.has_value() && *type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
         shape.has_value() && shape->empty();
}

std::optional<int64_t> ReadIotaCapacity(Ort::ConstValueInfo value) {
  auto type = GetTensorElementType(value);
  auto shape = GetTensorShape(value);
  if (!type.has_value() || *type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      !shape.has_value() || shape->size() != 3 || (*shape)[0] != 1) {
    return std::nullopt;
  }
  int64_t capacity = 0;
  if ((*shape)[1] > 0 && (*shape)[2] == 1) {
    capacity = (*shape)[1];
  } else if ((*shape)[1] == 1 && (*shape)[2] > 0) {
    capacity = (*shape)[2];
  } else {
    return std::nullopt;
  }
  auto values = ReadIntInitializerNoLimit(value);
  if (!values.has_value() || values->size() != static_cast<size_t>(capacity)) {
    return std::nullopt;
  }
  for (int64_t i = 0; i < capacity; ++i) {
    if ((*values)[static_cast<size_t>(i)] != i) {
      return std::nullopt;
    }
  }
  return capacity;
}

struct GenerateMTGRCustomMaskFusionCompute final : FusionNodeCompute {
  GenerateMTGRCustomMaskFusionCompute(std::vector<size_t> upstream_inputs,
                                      size_t target_input, size_t mask_output,
                                      std::optional<size_t> total_length_output,
                                      size_t target_length_output,
                                      int64_t capacity, int64_t prefix_length)
      : upstream_inputs(std::move(upstream_inputs)),
        target_input(target_input),
        mask_output(mask_output),
        total_length_output(total_length_output),
        target_length_output(target_length_output),
        capacity(capacity),
        prefix_length(prefix_length) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      int64_t upstream_length = prefix_length;
      for (size_t input_index : upstream_inputs) {
        auto shape =
            ctx.GetInput(input_index).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 3 || shape[1] < 0) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "Generate MTGR custom mask requires rank-3 upstream inputs");
        }
        if (upstream_length > std::numeric_limits<int64_t>::max() - shape[1]) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "Generate MTGR custom mask upstream length overflow");
        }
        upstream_length += shape[1];
      }

      auto target_shape =
          ctx.GetInput(target_input).GetTensorTypeAndShapeInfo().GetShape();
      if (target_shape.size() != 2 || target_shape[1] < 0) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "Generate MTGR custom mask requires a rank-2 target input");
      }
      const int64_t target_length = target_shape[1];
      if (upstream_length >
          std::numeric_limits<int64_t>::max() - target_length) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "Generate MTGR custom mask total length overflow");
      }

      Ort::UnownedValue target_length_value =
          ctx.GetOutput(target_length_output, {});
      if (!IsGpuMemory(target_length_value.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "Generate MTGR custom mask requires MUSA outputs");
      }
      if (target_length_value.GetTensorTypeAndShapeInfo().GetElementType() !=
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "Generate MTGR custom mask output dtype mismatch");
      }

      const int64_t total_length_value_host = upstream_length + target_length;
      if (!total_length_output.has_value()) {
        const int64_t dimension = std::min(total_length_value_host, capacity);
        if (dimension > 0 &&
            dimension > std::numeric_limits<int64_t>::max() / dimension /
                            static_cast<int64_t>(sizeof(int32_t))) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "Generate MTGR cropped mask output size overflow");
        }
        Ort::UnownedValue mask =
            ctx.GetOutput(mask_output, {1, 1, dimension, dimension});
        if (!IsGpuMemory(mask.GetTensorMemoryInfo()) ||
            mask.GetTensorTypeAndShapeInfo().GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "Generate MTGR cropped mask requires an int32 MUSA output");
        }
        return LaunchStatus(LaunchGenerateMTGRCroppedMaskKernel(
            mask.GetTensorMutableData<int32_t>(),
            target_length_value.GetTensorMutableData<int64_t>(),
            upstream_length, target_length, dimension, GetComputeStream(ctx)));
      }

      if (capacity > std::numeric_limits<int64_t>::max() / capacity) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "Generate MTGR full mask output size overflow");
      }
      Ort::UnownedValue mask =
          ctx.GetOutput(mask_output, {1, 1, capacity, capacity});
      Ort::UnownedValue total_length = ctx.GetOutput(*total_length_output, {});
      if (!IsGpuMemory(mask.GetTensorMemoryInfo()) ||
          !IsGpuMemory(total_length.GetTensorMemoryInfo()) ||
          mask.GetTensorTypeAndShapeInfo().GetElementType() !=
              ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL ||
          total_length.GetTensorTypeAndShapeInfo().GetElementType() !=
              ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "Generate MTGR full mask output type or memory mismatch");
      }
      return LaunchStatus(LaunchGenerateMTGRCustomMaskKernel(
          mask.GetTensorMutableData<bool>(),
          total_length.GetTensorMutableData<int64_t>(),
          target_length_value.GetTensorMutableData<int64_t>(), upstream_length,
          target_length, capacity, GetComputeStream(ctx)));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  std::vector<size_t> upstream_inputs;
  size_t target_input;
  size_t mask_output;
  std::optional<size_t> total_length_output;
  size_t target_length_output;
  int64_t capacity;
  int64_t prefix_length;
};

}  // namespace

bool IsGenerateMTGRCustomMaskFusionGraph(Ort::ConstGraph graph) {
  std::unordered_map<std::string, int> counts;
  int total = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    ++counts[node.GetOperatorType()];
    ++total;
  }
  const bool core = counts["Shape"] == 5 && counts["Gather"] == 5 &&
                    counts["Add"] == 6 && counts["Reshape"] == 2 &&
                    counts["Less"] == 4 && counts["GreaterOrEqual"] == 2 &&
                    counts["And"] == 8 && counts["Or"] == 2 &&
                    counts["Sub"] == 2 && counts["Cast"] == 2 &&
                    counts["Equal"] == 1 && counts["Unsqueeze"] == 1;
  const bool cropped_two_slice = total == 44 && counts["Cast"] == 3 &&
                                 counts["Unsqueeze"] == 2 &&
                                 counts["Slice"] == 2 && counts["Concat"] == 0;
  const bool cropped_one_slice = total == 44 && counts["Cast"] == 3 &&
                                 counts["Unsqueeze"] == 2 &&
                                 counts["Slice"] == 1 && counts["Concat"] == 1;
  return (total == 40 && core) ||
         ((cropped_two_slice || cropped_one_slice) && counts["Shape"] == 5 &&
          counts["Gather"] == 5 && counts["Add"] == 6 &&
          counts["Reshape"] == 2 && counts["Less"] == 4 &&
          counts["GreaterOrEqual"] == 2 && counts["And"] == 8 &&
          counts["Or"] == 2 && counts["Sub"] == 2 && counts["Equal"] == 1);
}

std::unique_ptr<FusionNodeCompute> CreateGenerateMTGRCustomMaskFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  if (!IsGenerateMTGRCustomMaskFusionGraph(graph)) {
    throw std::runtime_error("invalid Generate MTGR custom mask fusion graph");
  }

  const auto fused_inputs = fused_node.GetInputs();
  const auto fused_outputs = fused_node.GetOutputs();
  const auto input_indices = ValueIndices(fused_inputs);
  const auto output_indices = ValueIndices(fused_outputs);

  std::vector<size_t> upstream_inputs;
  std::optional<size_t> target_input;
  std::optional<size_t> mask_output;
  std::optional<size_t> total_length_output;
  std::optional<size_t> target_length_output;
  std::optional<int64_t> capacity;
  std::optional<int64_t> prefix_length;

  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Shape")) {
      auto inputs = node.GetInputs();
      if (inputs.size() != 1) {
        throw std::runtime_error("Generate MTGR custom mask invalid Shape");
      }
      auto shape = GetTensorShape(inputs[0]);
      if (!shape.has_value()) {
        throw std::runtime_error(
            "Generate MTGR custom mask input rank is unknown");
      }
      size_t index = RequiredIndex(input_indices, Name(inputs[0]), "input");
      if (shape->size() == 3) {
        upstream_inputs.push_back(index);
      } else if (shape->size() == 2) {
        if (target_input.has_value()) {
          throw std::runtime_error(
              "Generate MTGR custom mask has multiple target inputs");
        }
        target_input = index;
      } else {
        throw std::runtime_error(
            "Generate MTGR custom mask input rank mismatch");
      }
    } else if (IsOnnxOp(node, "Cast")) {
      auto outputs = node.GetOutputs();
      if (outputs.size() == 1 &&
          GetIntAttribute(node, "to").value_or(-1) ==
              ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
          output_indices.count(Name(outputs[0])) != 0) {
        mask_output = output_indices.at(Name(outputs[0]));
      }
    } else if (IsOnnxOp(node, "Unsqueeze")) {
      auto outputs = node.GetOutputs();
      if (!mask_output.has_value() && outputs.size() == 1 &&
          output_indices.count(Name(outputs[0])) != 0) {
        mask_output = output_indices.at(Name(outputs[0]));
      }
    } else if (IsOnnxOp(node, "Gather")) {
      auto inputs = node.GetInputs();
      auto outputs = node.GetOutputs();
      if (inputs.size() != 2 || outputs.size() != 1 ||
          output_indices.count(Name(outputs[0])) == 0) {
        continue;
      }
      Ort::ConstNode shape = inputs[0].GetProducerNode().node;
      if (IsOnnxOp(shape, "Shape") && shape.GetInputs().size() == 1) {
        auto source_shape = GetTensorShape(shape.GetInputs()[0]);
        if (source_shape.has_value() && source_shape->size() == 2) {
          target_length_output = output_indices.at(Name(outputs[0]));
        }
      }
    } else if (IsOnnxOp(node, "Add")) {
      auto inputs = node.GetInputs();
      auto outputs = node.GetOutputs();
      if (inputs.size() == 2) {
        Ort::ConstNode gather = inputs[0].GetProducerNode().node;
        auto candidate_prefix = ReadScalarIntInitializer(inputs[1]);
        if (IsOnnxOp(gather, "Gather") && candidate_prefix.has_value() &&
            *candidate_prefix >= 0 && gather.GetInputs().size() == 2) {
          Ort::ConstNode shape = gather.GetInputs()[0].GetProducerNode().node;
          if (IsOnnxOp(shape, "Shape") && shape.GetInputs().size() == 1) {
            auto source_shape = GetTensorShape(shape.GetInputs()[0]);
            if (source_shape.has_value() && source_shape->size() == 3) {
              if (prefix_length.has_value() &&
                  *prefix_length != *candidate_prefix) {
                throw std::runtime_error(
                    "Generate MTGR custom mask has inconsistent prefixes");
              }
              prefix_length = *candidate_prefix;
            }
          }
        }
      }
      if (outputs.size() == 1 && IsScalarInt64(outputs[0]) &&
          output_indices.count(Name(outputs[0])) != 0) {
        total_length_output = output_indices.at(Name(outputs[0]));
      }
    } else if (IsOnnxOp(node, "GreaterOrEqual")) {
      auto inputs = node.GetInputs();
      if (inputs.size() == 2) {
        auto candidate_capacity = ReadIotaCapacity(inputs[0]);
        if (candidate_capacity.has_value()) {
          if (capacity.has_value() && *capacity != *candidate_capacity) {
            throw std::runtime_error(
                "Generate MTGR custom mask has inconsistent capacities");
          }
          capacity = *candidate_capacity;
        }
      }
    }
  }

  if (upstream_inputs.size() != 4 || !target_input.has_value() ||
      !mask_output.has_value() || !target_length_output.has_value() ||
      !capacity.has_value() || !prefix_length.has_value() || *capacity <= 0 ||
      *prefix_length < 0 ||
      (graph.GetNodes().size() == 40 && !total_length_output.has_value()) ||
      (graph.GetNodes().size() != 40 && total_length_output.has_value())) {
    throw std::runtime_error(
        "Generate MTGR custom mask fused boundary is incomplete");
  }
  return std::make_unique<GenerateMTGRCustomMaskFusionCompute>(
      std::move(upstream_inputs), *target_input, *mask_output,
      total_length_output, *target_length_output, *capacity, *prefix_length);
}

}  // namespace musa_ep
