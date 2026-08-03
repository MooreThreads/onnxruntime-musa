// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

using ProducerMap = std::unordered_map<std::string, Ort::ConstNode>;

bool IsTensorType(Ort::ConstValueInfo value_info,
                  ONNXTensorElementDataType expected_type) {
  if (value_info == nullptr) {
    return false;
  }
  Ort::ConstTypeInfo type_info = value_info.TypeInfo();
  return type_info.GetONNXType() == ONNX_TYPE_TENSOR &&
         type_info.GetTensorTypeAndShapeInfo().GetElementType() ==
             expected_type;
}

bool IsScalarInitializer(Ort::ConstValueInfo value_info,
                         ONNXTensorElementDataType expected_type,
                         double expected_value, double tolerance) {
  if (!IsConstantInitializerValueInfo(value_info) ||
      !IsTensorType(value_info, expected_type)) {
    return false;
  }
  std::optional<float> value = ReadScalarFloatInitializer(value_info);
  return value.has_value() &&
         std::fabs(static_cast<double>(*value) - expected_value) <= tolerance;
}

bool IsCastTo(Ort::ConstNode node, ONNXTensorElementDataType type) {
  return IsOnnxOp(node, "Cast") &&
         GetIntAttribute(node, "to").value_or(-1) == static_cast<int64_t>(type);
}

int InputIndex(Ort::ConstNode node, const std::string& input_name) {
  std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (Name(inputs[i]) == input_name) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

bool HasExpectedConsumers(
    Ort::ConstValueInfo value_info,
    std::vector<std::pair<size_t, int64_t>> expected_consumers,
    const std::unordered_set<std::string>& graph_output_names) {
  if (value_info == nullptr ||
      graph_output_names.count(Name(value_info)) != 0) {
    return false;
  }
  std::vector<std::pair<size_t, int64_t>> actual_consumers;
  for (const auto& consumer : value_info.GetConsumers()) {
    actual_consumers.emplace_back(consumer.node.GetId(), consumer.index);
  }
  std::sort(actual_consumers.begin(), actual_consumers.end());
  std::sort(expected_consumers.begin(), expected_consumers.end());
  return actual_consumers == expected_consumers;
}

bool ShapesMatchCalibrationBoundary(Ort::ConstValueInfo task_relu,
                                    Ort::ConstValueInfo task_scores,
                                    Ort::ConstValueInfo tile_values,
                                    Ort::ConstValueInfo output,
                                    int64_t bucket_size) {
  auto relu_shape = GetTensorShape(task_relu);
  auto score_shape = GetTensorShape(task_scores);
  auto output_shape = GetTensorShape(output);
  if (!relu_shape.has_value() || !score_shape.has_value() ||
      !output_shape.has_value() || relu_shape->size() < 2 ||
      score_shape->size() != relu_shape->size() ||
      output_shape->size() != relu_shape->size() ||
      relu_shape->back() != bucket_size || score_shape->back() != 1 ||
      output_shape->back() != 1) {
    return false;
  }
  for (size_t i = 0; i + 1 < relu_shape->size(); ++i) {
    if (!KnownDimsEqual((*relu_shape)[i], (*score_shape)[i]) ||
        !KnownDimsEqual((*relu_shape)[i], (*output_shape)[i])) {
      return false;
    }
  }

  auto tile_shape = GetTensorShape(tile_values);
  if (tile_shape.has_value()) {
    if (tile_shape->size() != relu_shape->size()) {
      return false;
    }
    if (tile_shape->back() > 0 && tile_shape->back() != bucket_size) {
      return false;
    }
    for (size_t i = 0; i + 1 < tile_shape->size(); ++i) {
      if (!KnownDimsEqual((*tile_shape)[i], (*relu_shape)[i])) {
        return false;
      }
    }
  }
  return true;
}

bool CanFuseRecRankCalibration(
    Ort::ConstNode log_node, const ProducerMap& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(log_node, "Log") ||
      accepted_node_ids.count(log_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> log_inputs = log_node.GetInputs();
  std::vector<Ort::ConstValueInfo> log_outputs = log_node.GetOutputs();
  if (log_inputs.size() != 1 || log_outputs.size() != 1 ||
      !IsTensorType(log_outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    return false;
  }

  Ort::ConstNode div_node = FindProducer(producers, log_inputs[0]);
  if (!IsOnnxOp(div_node, "Div")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> div_inputs = div_node.GetInputs();
  std::vector<Ort::ConstValueInfo> div_outputs = div_node.GetOutputs();
  if (div_inputs.size() != 2 || div_outputs.size() != 1 ||
      !IsTensorType(div_outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    return false;
  }

  Ort::ConstNode probability_clip{nullptr};
  Ort::ConstNode complement_clip{nullptr};
  for (Ort::ConstValueInfo input : div_inputs) {
    Ort::ConstNode clip = FindProducer(producers, input);
    if (!IsOnnxOp(clip, "Clip")) {
      return false;
    }
    std::vector<Ort::ConstValueInfo> clip_inputs = clip.GetInputs();
    if (clip_inputs.size() != 3) {
      return false;
    }
    Ort::ConstNode data_producer = FindProducer(producers, clip_inputs[0]);
    if (IsOnnxOp(data_producer, "Cast")) {
      probability_clip = clip;
    } else if (IsOnnxOp(data_producer, "Sub")) {
      complement_clip = clip;
    } else {
      return false;
    }
  }
  if (!probability_clip || !complement_clip ||
      probability_clip.GetId() == complement_clip.GetId()) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> probability_clip_inputs =
      probability_clip.GetInputs();
  std::vector<Ort::ConstValueInfo> probability_clip_outputs =
      probability_clip.GetOutputs();
  std::vector<Ort::ConstValueInfo> complement_clip_inputs =
      complement_clip.GetInputs();
  std::vector<Ort::ConstValueInfo> complement_clip_outputs =
      complement_clip.GetOutputs();
  if (probability_clip_inputs.size() != 3 ||
      probability_clip_outputs.size() != 1 ||
      complement_clip_inputs.size() != 3 ||
      complement_clip_outputs.size() != 1 ||
      !IsScalarInitializer(probability_clip_inputs[1],
                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 1.0e-16,
                           1.0e-22) ||
      !IsScalarInitializer(probability_clip_inputs[2],
                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 1.0, 0.0) ||
      Name(probability_clip_inputs[1]) != Name(complement_clip_inputs[1]) ||
      Name(probability_clip_inputs[2]) != Name(complement_clip_inputs[2]) ||
      !IsTensorType(probability_clip_outputs[0],
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      !IsTensorType(complement_clip_outputs[0],
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    return false;
  }

  Ort::ConstNode complement_sub =
      FindProducer(producers, complement_clip_inputs[0]);
  std::vector<Ort::ConstValueInfo> complement_sub_inputs =
      complement_sub ? complement_sub.GetInputs()
                     : std::vector<Ort::ConstValueInfo>{};
  std::vector<Ort::ConstValueInfo> complement_sub_outputs =
      complement_sub ? complement_sub.GetOutputs()
                     : std::vector<Ort::ConstValueInfo>{};
  if (!IsOnnxOp(complement_sub, "Sub") || complement_sub_inputs.size() != 2 ||
      complement_sub_outputs.size() != 1 ||
      Name(complement_sub_inputs[0]) != Name(probability_clip_inputs[2]) ||
      !IsTensorType(complement_sub_outputs[0],
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    return false;
  }

  Ort::ConstNode output_cast =
      FindProducer(producers, probability_clip_inputs[0]);
  std::vector<Ort::ConstValueInfo> output_cast_inputs =
      output_cast ? output_cast.GetInputs()
                  : std::vector<Ort::ConstValueInfo>{};
  std::vector<Ort::ConstValueInfo> output_cast_outputs =
      output_cast ? output_cast.GetOutputs()
                  : std::vector<Ort::ConstValueInfo>{};
  if (!IsCastTo(output_cast, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      output_cast_inputs.size() != 1 || output_cast_outputs.size() != 1) {
    return false;
  }
  const bool complement_from_clipped_probability =
      Name(complement_sub_inputs[1]) == Name(probability_clip_outputs[0]);
  const bool complement_from_raw_probability =
      Name(complement_sub_inputs[1]) == Name(output_cast_outputs[0]);
  if (!complement_from_clipped_probability &&
      !complement_from_raw_probability) {
    return false;
  }

  Ort::ConstNode reduce_node = FindProducer(producers, output_cast_inputs[0]);
  std::vector<Ort::ConstValueInfo> reduce_inputs =
      reduce_node ? reduce_node.GetInputs()
                  : std::vector<Ort::ConstValueInfo>{};
  std::vector<Ort::ConstValueInfo> reduce_outputs =
      reduce_node ? reduce_node.GetOutputs()
                  : std::vector<Ort::ConstValueInfo>{};
  if (!IsOnnxOp(reduce_node, "ReduceSum") || reduce_inputs.size() != 2 ||
      reduce_outputs.size() != 1 ||
      GetIntAttribute(reduce_node, "keepdims").value_or(1) != 1 ||
      !ReadSmallIntInitializer(reduce_inputs[1]).has_value() ||
      *ReadSmallIntInitializer(reduce_inputs[1]) != std::vector<int64_t>{-1} ||
      !IsTensorType(reduce_outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)) {
    return false;
  }

  Ort::ConstNode normalized_mul = FindProducer(producers, reduce_inputs[0]);
  std::vector<Ort::ConstValueInfo> normalized_mul_inputs =
      normalized_mul ? normalized_mul.GetInputs()
                     : std::vector<Ort::ConstValueInfo>{};
  std::vector<Ort::ConstValueInfo> normalized_mul_outputs =
      normalized_mul ? normalized_mul.GetOutputs()
                     : std::vector<Ort::ConstValueInfo>{};
  if (!IsOnnxOp(normalized_mul, "Mul") || normalized_mul_inputs.size() != 2 ||
      normalized_mul_outputs.size() != 1) {
    return false;
  }
  Ort::ConstNode weighted_mul{nullptr};
  Ort::ConstValueInfo inverse_bucket{nullptr};
  for (Ort::ConstValueInfo input : normalized_mul_inputs) {
    Ort::ConstNode producer = FindProducer(producers, input);
    if (IsOnnxOp(producer, "Mul")) {
      weighted_mul = producer;
    } else if (IsConstantInitializerValueInfo(input)) {
      inverse_bucket = input;
    }
  }
  if (!weighted_mul || inverse_bucket == nullptr ||
      !IsTensorType(inverse_bucket, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> weighted_mul_inputs =
      weighted_mul.GetInputs();
  std::vector<Ort::ConstValueInfo> weighted_mul_outputs =
      weighted_mul.GetOutputs();
  if (weighted_mul_inputs.size() != 2 || weighted_mul_outputs.size() != 1) {
    return false;
  }
  Ort::ConstNode mask_add{nullptr};
  Ort::ConstValueInfo task_relu{nullptr};
  for (Ort::ConstValueInfo input : weighted_mul_inputs) {
    Ort::ConstNode producer = FindProducer(producers, input);
    if (IsOnnxOp(producer, "Add")) {
      mask_add = producer;
    } else {
      task_relu = input;
    }
  }
  if (!mask_add || task_relu == nullptr ||
      !IsTensorType(task_relu, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> mask_add_inputs = mask_add.GetInputs();
  std::vector<Ort::ConstValueInfo> mask_add_outputs = mask_add.GetOutputs();
  if (mask_add_inputs.size() != 2 || mask_add_outputs.size() != 1) {
    return false;
  }
  Ort::ConstNode less_cast{nullptr};
  Ort::ConstNode fractional_mul{nullptr};
  for (Ort::ConstValueInfo input : mask_add_inputs) {
    Ort::ConstNode producer = FindProducer(producers, input);
    if (IsOnnxOp(producer, "Cast")) {
      less_cast = producer;
    } else if (IsOnnxOp(producer, "Mul")) {
      fractional_mul = producer;
    }
  }
  if (!IsCastTo(less_cast, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) ||
      !fractional_mul) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> fractional_mul_inputs =
      fractional_mul.GetInputs();
  std::vector<Ort::ConstValueInfo> fractional_mul_outputs =
      fractional_mul.GetOutputs();
  if (fractional_mul_inputs.size() != 2 || fractional_mul_outputs.size() != 1) {
    return false;
  }
  Ort::ConstNode equal_cast{nullptr};
  Ort::ConstNode fractional_sub{nullptr};
  for (Ort::ConstValueInfo input : fractional_mul_inputs) {
    Ort::ConstNode producer = FindProducer(producers, input);
    if (IsOnnxOp(producer, "Cast")) {
      equal_cast = producer;
    } else if (IsOnnxOp(producer, "Sub")) {
      fractional_sub = producer;
    }
  }
  if (!IsCastTo(equal_cast, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) ||
      !fractional_sub) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> less_cast_inputs = less_cast.GetInputs();
  std::vector<Ort::ConstValueInfo> less_cast_outputs = less_cast.GetOutputs();
  std::vector<Ort::ConstValueInfo> equal_cast_inputs = equal_cast.GetInputs();
  std::vector<Ort::ConstValueInfo> equal_cast_outputs = equal_cast.GetOutputs();
  if (less_cast_inputs.size() != 1 || less_cast_outputs.size() != 1 ||
      equal_cast_inputs.size() != 1 || equal_cast_outputs.size() != 1) {
    return false;
  }
  Ort::ConstNode less_node = FindProducer(producers, less_cast_inputs[0]);
  Ort::ConstNode equal_node = FindProducer(producers, equal_cast_inputs[0]);
  if (!IsOnnxOp(less_node, "Less") || !IsOnnxOp(equal_node, "Equal")) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> less_inputs = less_node.GetInputs();
  std::vector<Ort::ConstValueInfo> less_outputs = less_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> equal_inputs = equal_node.GetInputs();
  std::vector<Ort::ConstValueInfo> equal_outputs = equal_node.GetOutputs();
  if (less_inputs.size() != 2 || less_outputs.size() != 1 ||
      equal_inputs.size() != 2 || equal_outputs.size() != 1) {
    return false;
  }
  Ort::ConstNode floor_node = FindProducer(producers, less_inputs[1]);
  if (!IsOnnxOp(floor_node, "Floor")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> floor_inputs = floor_node.GetInputs();
  std::vector<Ort::ConstValueInfo> floor_outputs = floor_node.GetOutputs();
  if (floor_inputs.size() != 1 || floor_outputs.size() != 1 ||
      Name(equal_inputs[0]) != Name(less_inputs[0]) ||
      Name(equal_inputs[1]) != Name(floor_outputs[0])) {
    return false;
  }
  Ort::ConstValueInfo tile_values = less_inputs[0];

  std::vector<Ort::ConstValueInfo> fractional_sub_inputs =
      fractional_sub.GetInputs();
  std::vector<Ort::ConstValueInfo> fractional_sub_outputs =
      fractional_sub.GetOutputs();
  if (fractional_sub_inputs.size() != 2 || fractional_sub_outputs.size() != 1 ||
      Name(fractional_sub_inputs[0]) != Name(floor_inputs[0]) ||
      Name(fractional_sub_inputs[1]) != Name(floor_outputs[0])) {
    return false;
  }

  Ort::ConstNode scaled_mul = FindProducer(producers, floor_inputs[0]);
  std::vector<Ort::ConstValueInfo> scaled_mul_inputs =
      scaled_mul ? scaled_mul.GetInputs() : std::vector<Ort::ConstValueInfo>{};
  std::vector<Ort::ConstValueInfo> scaled_mul_outputs =
      scaled_mul ? scaled_mul.GetOutputs() : std::vector<Ort::ConstValueInfo>{};
  if (!IsOnnxOp(scaled_mul, "Mul") || scaled_mul_inputs.size() != 2 ||
      scaled_mul_outputs.size() != 1) {
    return false;
  }
  Ort::ConstValueInfo task_scores{nullptr};
  Ort::ConstValueInfo bucket_initializer{nullptr};
  for (Ort::ConstValueInfo input : scaled_mul_inputs) {
    if (IsConstantInitializerValueInfo(input)) {
      bucket_initializer = input;
    } else {
      task_scores = input;
    }
  }
  std::optional<float> bucket_value =
      ReadScalarFloatInitializer(bucket_initializer);
  if (task_scores == nullptr || bucket_initializer == nullptr ||
      !IsTensorType(task_scores, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) ||
      !IsTensorType(tile_values, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) ||
      !IsTensorType(bucket_initializer, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) ||
      !bucket_value.has_value() || !std::isfinite(*bucket_value)) {
    return false;
  }
  const int64_t bucket_size = static_cast<int64_t>(std::llround(*bucket_value));
  if (bucket_size <= 0 ||
      std::fabs(static_cast<double>(*bucket_value) -
                static_cast<double>(bucket_size)) > 0.0 ||
      bucket_size > std::numeric_limits<int32_t>::max() ||
      !IsScalarInitializer(inverse_bucket, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
                           1.0 / static_cast<double>(bucket_size), 1.0e-9) ||
      !ShapesMatchCalibrationBoundary(task_relu, task_scores, tile_values,
                                      log_outputs[0], bucket_size)) {
    return false;
  }

  const int scaled_to_floor =
      InputIndex(floor_node, Name(scaled_mul_outputs[0]));
  const int scaled_to_sub =
      InputIndex(fractional_sub, Name(scaled_mul_outputs[0]));
  const int floor_to_less = InputIndex(less_node, Name(floor_outputs[0]));
  const int floor_to_equal = InputIndex(equal_node, Name(floor_outputs[0]));
  const int floor_to_sub = InputIndex(fractional_sub, Name(floor_outputs[0]));
  const int probability_to_div =
      InputIndex(div_node, Name(probability_clip_outputs[0]));
  const int complement_to_div =
      InputIndex(div_node, Name(complement_clip_outputs[0]));
  if (scaled_to_floor < 0 || scaled_to_sub < 0 || floor_to_less != 1 ||
      floor_to_equal < 0 || floor_to_sub < 0 || probability_to_div < 0 ||
      complement_to_div < 0 ||
      !HasExpectedConsumers(scaled_mul_outputs[0],
                            {{floor_node.GetId(), scaled_to_floor},
                             {fractional_sub.GetId(), scaled_to_sub}},
                            graph_output_names) ||
      !HasExpectedConsumers(floor_outputs[0],
                            {{less_node.GetId(), floor_to_less},
                             {equal_node.GetId(), floor_to_equal},
                             {fractional_sub.GetId(), floor_to_sub}},
                            graph_output_names) ||
      !HasExpectedConsumers(less_outputs[0], {{less_cast.GetId(), 0}},
                            graph_output_names) ||
      !HasExpectedConsumers(equal_outputs[0], {{equal_cast.GetId(), 0}},
                            graph_output_names) ||
      !HasExpectedConsumers(less_cast_outputs[0], {{mask_add.GetId(), 0}},
                            graph_output_names) ||
      !HasExpectedConsumers(equal_cast_outputs[0],
                            {{fractional_mul.GetId(), 0}},
                            graph_output_names) ||
      !HasExpectedConsumers(fractional_sub_outputs[0],
                            {{fractional_mul.GetId(), 1}},
                            graph_output_names) ||
      !HasExpectedConsumers(fractional_mul_outputs[0], {{mask_add.GetId(), 1}},
                            graph_output_names) ||
      !HasExpectedConsumers(mask_add_outputs[0], {{weighted_mul.GetId(), 1}},
                            graph_output_names) ||
      !HasExpectedConsumers(weighted_mul_outputs[0],
                            {{normalized_mul.GetId(), 0}},
                            graph_output_names) ||
      !HasExpectedConsumers(normalized_mul_outputs[0],
                            {{reduce_node.GetId(), 0}}, graph_output_names) ||
      !HasExpectedConsumers(reduce_outputs[0], {{output_cast.GetId(), 0}},
                            graph_output_names) ||
      !HasExpectedConsumers(
          output_cast_outputs[0],
          complement_from_clipped_probability
              ? std::vector<std::pair<size_t, int64_t>>{{probability_clip
                                                             .GetId(),
                                                         0}}
              : std::vector<
                    std::pair<size_t, int64_t>>{{probability_clip.GetId(), 0},
                                                {complement_sub.GetId(), 1}},
          graph_output_names) ||
      !HasExpectedConsumers(
          probability_clip_outputs[0],
          complement_from_clipped_probability
              ? std::vector<std::pair<size_t, int64_t>>{{complement_sub.GetId(),
                                                         1},
                                                        {div_node.GetId(),
                                                         probability_to_div}}
              : std::vector<std::pair<size_t, int64_t>>{{div_node.GetId(),
                                                         probability_to_div}},
          graph_output_names) ||
      !HasExpectedConsumers(complement_sub_outputs[0],
                            {{complement_clip.GetId(), 0}},
                            graph_output_names) ||
      !HasExpectedConsumers(complement_clip_outputs[0],
                            {{div_node.GetId(), complement_to_div}},
                            graph_output_names) ||
      !HasExpectedConsumers(div_outputs[0], {{log_node.GetId(), 0}},
                            graph_output_names)) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  for (Ort::ConstNode node :
       {scaled_mul, floor_node, less_node, less_cast, equal_node, equal_cast,
        fractional_sub, fractional_mul, mask_add, weighted_mul, normalized_mul,
        reduce_node, output_cast, probability_clip, complement_sub,
        complement_clip, div_node, log_node}) {
    if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }
  if (fusion_nodes.size() != 18 || !FusionHasNoExternalPathBetweenSelectedNodes(
                                       fusion_nodes, selected_node_ids)) {
    return false;
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindRecRankCalibrationFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  ProducerMap producers = BuildProducerMap(all_nodes);
  for (Ort::ConstNode log_node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (CanFuseRecRankCalibration(log_node, producers, graph_output_names,
                                  accepted_node_ids, fusion_nodes)) {
      fusions.push_back(std::move(fusion_nodes));
    }
  }
  return fusions;
}

}  // namespace musa_ep
