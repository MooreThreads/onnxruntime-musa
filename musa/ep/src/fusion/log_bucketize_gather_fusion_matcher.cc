// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {
bool CastTo(Ort::ConstNode cast_node, int64_t expected_to) {
  return GetIntAttribute(cast_node, "to").value_or(-1) == expected_to;
}

bool IsPayloadTensor(Ort::ConstValueInfo value_info) {
  auto elem_type = GetTensorElementType(value_info);
  if (!elem_type.has_value()) {
    return false;
  }
  return *elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         *elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         *elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
}

bool HasSamePayloadType(Ort::ConstValueInfo lhs, Ort::ConstValueInfo rhs) {
  auto lhs_type = GetTensorElementType(lhs);
  auto rhs_type = GetTensorElementType(rhs);
  return lhs_type.has_value() && rhs_type.has_value() &&
         *lhs_type == *rhs_type && IsPayloadTensor(lhs);
}

bool HasSamePayloadTypeIfKnown(Ort::ConstValueInfo lhs,
                               Ort::ConstValueInfo rhs) {
  auto lhs_type = GetTensorElementType(lhs);
  auto rhs_type = GetTensorElementType(rhs);
  if (!lhs_type.has_value() || !rhs_type.has_value()) {
    return true;
  }
  return *lhs_type == *rhs_type && IsPayloadTensor(lhs);
}

bool HasPayloadTypeIfKnown(Ort::ConstValueInfo value_info) {
  auto elem_type = GetTensorElementType(value_info);
  return !elem_type.has_value() || IsPayloadTensor(value_info);
}

bool IsIntTensorOrProducedByIntCast(
    Ort::ConstValueInfo value_info,
    const std::unordered_map<std::string, Ort::ConstNode>& producers) {
  if (IsIntTensorValueInfo(value_info)) {
    return true;
  }
  Ort::ConstNode producer = FindProducer(producers, value_info);
  return IsOnnxOp(producer, "Cast") &&
         CastTo(producer, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
}

bool HasNoExternalConsumers(
    Ort::ConstValueInfo value_info,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& allowed_consumer_ids) {
  if (value_info == nullptr ||
      graph_output_names.count(Name(value_info)) != 0) {
    return false;
  }
  for (const auto& consumer : value_info.GetConsumers()) {
    if (allowed_consumer_ids.count(consumer.node.GetId()) == 0) {
      return false;
    }
  }
  return true;
}

bool AddNodeIfPresent(Ort::ConstNode node,
                      const std::unordered_set<size_t>& accepted_node_ids,
                      std::unordered_set<size_t>& selected_node_ids,
                      std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!node) {
    return true;
  }
  return AddFusionNode(node, accepted_node_ids, selected_node_ids,
                       fusion_nodes);
}

bool AddConstantProducerChain(
    Ort::ConstValueInfo value_info,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::unordered_set<size_t>& selected_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  Ort::ConstNode producer = FindProducer(producers, value_info);
  if (!producer) {
    return true;
  }
  if (IsOnnxOp(producer, "Constant")) {
    return AddFusionNode(producer, accepted_node_ids, selected_node_ids,
                         fusion_nodes);
  }
  if (IsOnnxOp(producer, "Cast")) {
    if (!AddFusionNode(producer, accepted_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
    std::vector<Ort::ConstValueInfo> inputs = producer.GetInputs();
    return inputs.size() == 1 &&
           AddConstantProducerChain(inputs[0], producers, accepted_node_ids,
                                    selected_node_ids, fusion_nodes);
  }
  return true;
}

std::optional<std::vector<int64_t>> ReadSmallIntConstantOrInitializer(
    Ort::ConstValueInfo value_info,
    const std::unordered_map<std::string, Ort::ConstNode>& producers) {
  std::optional<std::vector<int64_t>> initializer =
      ReadSmallIntInitializer(value_info);
  if (initializer.has_value()) {
    return initializer;
  }

  Ort::ConstNode producer = FindProducer(producers, value_info);
  if (!IsOnnxOp(producer, "Constant")) {
    return std::nullopt;
  }
  Ort::ConstOpAttr attr;
  Ort::Status attr_status = producer.GetAttributeByName("value", attr);
  if (!attr_status.IsOK()) {
    return std::nullopt;
  }
  Ort::Value value{nullptr};
  Ort::Status value_status = attr.GetTensorAttributeAsOrtValue(value);
  if (!value_status.IsOK() || !value) {
    return std::nullopt;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  const size_t count = static_cast<size_t>(info.GetElementCount());
  if (count > kSmallInitializerThreshold) {
    return std::nullopt;
  }

  std::vector<int64_t> result;
  result.reserve(count);
  const ONNXTensorElementDataType elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t* data = value.GetTensorData<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      result.push_back(static_cast<int64_t>(data[i]));
    }
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* data = value.GetTensorData<int64_t>();
    result.assign(data, data + count);
  } else {
    return std::nullopt;
  }
  return result;
}

bool CanFuseLogBucketizeGather(
    Ort::ConstNode output_anchor,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  Ort::ConstNode cast_node{nullptr};
  Ort::ConstNode unsqueeze_node{nullptr};
  if (IsOnnxOp(output_anchor, "Cast")) {
    if (!CastTo(output_anchor, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)) {
      return false;
    }
    std::vector<Ort::ConstValueInfo> cast_inputs = output_anchor.GetInputs();
    std::vector<Ort::ConstValueInfo> cast_outputs = output_anchor.GetOutputs();
    if (cast_inputs.size() != 1 || cast_outputs.size() != 1 ||
        !HasSamePayloadTypeIfKnown(cast_inputs[0], cast_outputs[0])) {
      return false;
    }
    unsqueeze_node = FindProducer(producers, cast_inputs[0]);
    if (!IsOnnxOp(unsqueeze_node, "Unsqueeze")) {
      return false;
    }
    cast_node = output_anchor;
  } else if (IsOnnxOp(output_anchor, "Unsqueeze")) {
    std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
        output_anchor.GetOutputs();
    if (unsqueeze_outputs.size() != 1) {
      return false;
    }
    for (const auto& consumer : unsqueeze_outputs[0].GetConsumers()) {
      if (IsOnnxOp(consumer.node, "Cast") &&
          CastTo(consumer.node, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)) {
        return false;
      }
    }
    unsqueeze_node = output_anchor;
  } else {
    return false;
  }

  if (accepted_node_ids.count(unsqueeze_node.GetId()) != 0 ||
      (cast_node && accepted_node_ids.count(cast_node.GetId()) != 0)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
      unsqueeze_node.GetInputs();
  std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
      unsqueeze_node.GetOutputs();
  if (unsqueeze_inputs.size() != 2 || unsqueeze_outputs.size() != 1 ||
      ReadSmallIntConstantOrInitializer(unsqueeze_inputs[1], producers)
              .value_or(std::vector<int64_t>{})
              .size() != 1) {
    return false;
  }

  Ort::ConstNode gather_node = FindProducer(producers, unsqueeze_inputs[0]);
  if (!IsOnnxOp(gather_node, "Gather") ||
      accepted_node_ids.count(gather_node.GetId()) != 0 ||
      GetIntAttribute(gather_node, "axis").value_or(0) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gather_outputs = gather_node.GetOutputs();
  if (gather_inputs.size() != 2 || gather_outputs.size() != 1 ||
      !HasOnlyConsumer(gather_outputs[0], unsqueeze_node, 0) ||
      !IsPayloadTensor(gather_inputs[0]) ||
      !HasPayloadTypeIfKnown(gather_outputs[0]) ||
      !HasSamePayloadTypeIfKnown(gather_inputs[0], gather_outputs[0]) ||
      !HasSamePayloadTypeIfKnown(gather_outputs[0], unsqueeze_outputs[0])) {
    return false;
  }
  auto table_shape = GetTensorShape(gather_inputs[0]);
  if (!table_shape.has_value() || table_shape->empty() ||
      (*table_shape)[0] < 127) {
    return false;
  }

  Ort::ConstNode low_where = FindProducer(producers, gather_inputs[1]);
  if (!IsOnnxOp(low_where, "Where") ||
      accepted_node_ids.count(low_where.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> low_where_inputs = low_where.GetInputs();
  std::vector<Ort::ConstValueInfo> low_where_outputs = low_where.GetOutputs();
  if (low_where_inputs.size() != 3 || low_where_outputs.size() != 1 ||
      !HasOnlyConsumer(low_where_outputs[0], gather_node, 1)) {
    return false;
  }

  Ort::ConstNode less_equal = FindProducer(producers, low_where_inputs[0]);
  Ort::ConstNode high_where = FindProducer(producers, low_where_inputs[2]);
  if (!IsOnnxOp(less_equal, "LessOrEqual") || !IsOnnxOp(high_where, "Where") ||
      accepted_node_ids.count(less_equal.GetId()) != 0 ||
      accepted_node_ids.count(high_where.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> less_inputs = less_equal.GetInputs();
  std::vector<Ort::ConstValueInfo> less_outputs = less_equal.GetOutputs();
  std::vector<Ort::ConstValueInfo> high_where_inputs = high_where.GetInputs();
  std::vector<Ort::ConstValueInfo> high_where_outputs = high_where.GetOutputs();
  if (less_inputs.size() != 2 || less_outputs.size() != 1 ||
      high_where_inputs.size() != 3 || high_where_outputs.size() != 1 ||
      !HasOnlyConsumer(less_outputs[0], low_where, 0) ||
      !HasOnlyConsumer(high_where_outputs[0], low_where, 2)) {
    return false;
  }

  Ort::ConstNode greater_equal = FindProducer(producers, high_where_inputs[0]);
  Ort::ConstNode index_clip = FindProducer(producers, high_where_inputs[2]);
  if (!IsOnnxOp(greater_equal, "GreaterOrEqual") ||
      !IsOnnxOp(index_clip, "Clip") ||
      accepted_node_ids.count(greater_equal.GetId()) != 0 ||
      accepted_node_ids.count(index_clip.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> greater_inputs = greater_equal.GetInputs();
  std::vector<Ort::ConstValueInfo> greater_outputs = greater_equal.GetOutputs();
  if (greater_inputs.size() != 2 || greater_outputs.size() != 1 ||
      !HasOnlyConsumer(greater_outputs[0], high_where, 0) ||
      Name(greater_inputs[0]) != Name(less_inputs[0])) {
    return false;
  }
  Ort::ConstValueInfo log_bucket = less_inputs[0];

  Ort::ConstNode log_div = FindProducer(producers, log_bucket);
  if (!IsOnnxOp(log_div, "Div") ||
      accepted_node_ids.count(log_div.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> log_div_inputs = log_div.GetInputs();
  std::vector<Ort::ConstValueInfo> log_div_outputs = log_div.GetOutputs();
  if (log_div_inputs.size() != 2 || log_div_outputs.size() != 1) {
    return false;
  }

  Ort::ConstNode log_node = FindProducer(producers, log_div_inputs[0]);
  if (!IsOnnxOp(log_node, "Log") ||
      accepted_node_ids.count(log_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> log_inputs = log_node.GetInputs();
  std::vector<Ort::ConstValueInfo> log_outputs = log_node.GetOutputs();
  if (log_inputs.size() != 1 || log_outputs.size() != 1 ||
      !HasOnlyConsumer(log_outputs[0], log_div, 0)) {
    return false;
  }

  Ort::ConstNode value_clip = FindProducer(producers, log_inputs[0]);
  if (!IsOnnxOp(value_clip, "Clip") ||
      accepted_node_ids.count(value_clip.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> value_clip_inputs = value_clip.GetInputs();
  std::vector<Ort::ConstValueInfo> value_clip_outputs = value_clip.GetOutputs();
  if (value_clip_inputs.size() < 2 || value_clip_outputs.size() != 1 ||
      !HasOnlyConsumer(value_clip_outputs[0], log_node, 0)) {
    return false;
  }

  Ort::ConstNode time_div = FindProducer(producers, value_clip_inputs[0]);
  if (!IsOnnxOp(time_div, "Div") ||
      accepted_node_ids.count(time_div.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> time_div_inputs = time_div.GetInputs();
  std::vector<Ort::ConstValueInfo> time_div_outputs = time_div.GetOutputs();
  if (time_div_inputs.size() != 2 || time_div_outputs.size() != 1 ||
      !HasOnlyConsumer(time_div_outputs[0], value_clip, 0)) {
    return false;
  }

  Ort::ConstNode diff_cast = FindProducer(producers, time_div_inputs[0]);
  if (!IsOnnxOp(diff_cast, "Cast") ||
      !CastTo(diff_cast, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      accepted_node_ids.count(diff_cast.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> diff_cast_inputs = diff_cast.GetInputs();
  std::vector<Ort::ConstValueInfo> diff_cast_outputs = diff_cast.GetOutputs();
  if (diff_cast_inputs.size() != 1 || diff_cast_outputs.size() != 1 ||
      !HasOnlyConsumer(diff_cast_outputs[0], time_div, 0)) {
    return false;
  }

  Ort::ConstNode diff_clip = FindProducer(producers, diff_cast_inputs[0]);
  if (!IsOnnxOp(diff_clip, "Clip") ||
      accepted_node_ids.count(diff_clip.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> diff_clip_inputs = diff_clip.GetInputs();
  std::vector<Ort::ConstValueInfo> diff_clip_outputs = diff_clip.GetOutputs();
  if (diff_clip_inputs.size() < 2 || diff_clip_outputs.size() != 1 ||
      !HasOnlyConsumer(diff_clip_outputs[0], diff_cast, 0)) {
    return false;
  }

  Ort::ConstNode sub_node = FindProducer(producers, diff_clip_inputs[0]);
  if (!IsOnnxOp(sub_node, "Sub") ||
      accepted_node_ids.count(sub_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> sub_inputs = sub_node.GetInputs();
  std::vector<Ort::ConstValueInfo> sub_outputs = sub_node.GetOutputs();
  if (sub_inputs.size() != 2 || sub_outputs.size() != 1 ||
      !HasOnlyConsumer(sub_outputs[0], diff_clip, 0) ||
      !IsIntTensorOrProducedByIntCast(sub_inputs[0], producers)) {
    return false;
  }

  Ort::ConstNode sequence_cast{nullptr};
  Ort::ConstValueInfo sequence_value = sub_inputs[1];
  sequence_cast = FindProducer(producers, sequence_value);
  if (IsOnnxOp(sequence_cast, "Cast") &&
      CastTo(sequence_cast, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    std::vector<Ort::ConstValueInfo> cast_inputs = sequence_cast.GetInputs();
    std::vector<Ort::ConstValueInfo> cast_outputs = sequence_cast.GetOutputs();
    if (accepted_node_ids.count(sequence_cast.GetId()) != 0 ||
        cast_inputs.size() != 1 || cast_outputs.size() != 1 ||
        !HasOnlyConsumer(cast_outputs[0], sub_node, 1)) {
      return false;
    }
    sequence_value = cast_inputs[0];
  } else {
    sequence_cast = Ort::ConstNode{nullptr};
  }

  Ort::ConstNode sequence_reshape = FindProducer(producers, sequence_value);
  if (!IsOnnxOp(sequence_reshape, "Reshape") ||
      accepted_node_ids.count(sequence_reshape.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> reshape_inputs =
      sequence_reshape.GetInputs();
  std::vector<Ort::ConstValueInfo> reshape_outputs =
      sequence_reshape.GetOutputs();
  if (reshape_inputs.size() != 2 || reshape_outputs.size() != 1 ||
      !IsIntTensorOrProducedByIntCast(reshape_inputs[0], producers)) {
    return false;
  }
  if (sequence_cast) {
    if (!HasOnlyConsumer(reshape_outputs[0], sequence_cast, 0)) {
      return false;
    }
  } else if (!HasOnlyConsumer(reshape_outputs[0], sub_node, 1)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> index_clip_inputs = index_clip.GetInputs();
  std::vector<Ort::ConstValueInfo> index_clip_outputs = index_clip.GetOutputs();
  if (index_clip_inputs.size() < 3 || index_clip_outputs.size() != 1) {
    return false;
  }
  Ort::ConstNode add_node = FindProducer(producers, index_clip_inputs[0]);
  if (!IsOnnxOp(add_node, "Add") ||
      accepted_node_ids.count(add_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  if (add_inputs.size() != 2 || add_outputs.size() != 1 ||
      !HasOnlyConsumer(add_outputs[0], index_clip, 0)) {
    return false;
  }
  Ort::ConstNode floor_cast = FindProducer(producers, add_inputs[0]);
  if (!IsOnnxOp(floor_cast, "Cast") ||
      !CastTo(floor_cast, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
      accepted_node_ids.count(floor_cast.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> floor_cast_inputs = floor_cast.GetInputs();
  std::vector<Ort::ConstValueInfo> floor_cast_outputs = floor_cast.GetOutputs();
  if (floor_cast_inputs.size() != 1 || floor_cast_outputs.size() != 1 ||
      !HasOnlyConsumer(floor_cast_outputs[0], add_node, 0)) {
    return false;
  }
  Ort::ConstNode floor_node = FindProducer(producers, floor_cast_inputs[0]);
  if (!IsOnnxOp(floor_node, "Floor") ||
      accepted_node_ids.count(floor_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> floor_inputs = floor_node.GetInputs();
  std::vector<Ort::ConstValueInfo> floor_outputs = floor_node.GetOutputs();
  if (floor_inputs.size() != 1 || floor_outputs.size() != 1 ||
      !HasOnlyConsumer(floor_outputs[0], floor_cast, 0)) {
    return false;
  }

  Ort::ConstNode scale_mul = FindProducer(producers, floor_inputs[0]);
  if (!IsOnnxOp(scale_mul, "Mul") ||
      accepted_node_ids.count(scale_mul.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> scale_mul_inputs = scale_mul.GetInputs();
  std::vector<Ort::ConstValueInfo> scale_mul_outputs = scale_mul.GetOutputs();
  if (scale_mul_inputs.size() != 2 || scale_mul_outputs.size() != 1 ||
      !HasOnlyConsumer(scale_mul_outputs[0], floor_node, 0)) {
    return false;
  }
  Ort::ConstNode span_div = FindProducer(producers, scale_mul_inputs[0]);
  if (!IsOnnxOp(span_div, "Div") ||
      accepted_node_ids.count(span_div.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> span_div_inputs = span_div.GetInputs();
  std::vector<Ort::ConstValueInfo> span_div_outputs = span_div.GetOutputs();
  if (span_div_inputs.size() != 2 || span_div_outputs.size() != 1 ||
      !HasOnlyConsumer(span_div_outputs[0], scale_mul, 0)) {
    return false;
  }
  Ort::ConstNode start_sub = FindProducer(producers, span_div_inputs[0]);
  if (!IsOnnxOp(start_sub, "Sub") ||
      accepted_node_ids.count(start_sub.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> start_sub_inputs = start_sub.GetInputs();
  std::vector<Ort::ConstValueInfo> start_sub_outputs = start_sub.GetOutputs();
  if (start_sub_inputs.size() != 2 || start_sub_outputs.size() != 1 ||
      Name(start_sub_inputs[0]) != Name(log_bucket) ||
      !HasOnlyConsumer(start_sub_outputs[0], span_div, 0)) {
    return false;
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  std::vector<Ort::ConstNode> required_nodes = {
      sequence_reshape, sub_node,   diff_clip,  diff_cast, time_div,
      value_clip,       log_node,   log_div,    start_sub, span_div,
      scale_mul,        floor_node, floor_cast, add_node,  index_clip,
      greater_equal,    high_where, less_equal, low_where, gather_node,
      unsqueeze_node};
  if (sequence_cast) {
    required_nodes.push_back(sequence_cast);
  }
  if (cast_node) {
    required_nodes.push_back(cast_node);
  }
  for (Ort::ConstNode node : required_nodes) {
    if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }

  const auto add_fill_shape_chain = [&](Ort::ConstValueInfo fill_value) {
    Ort::ConstNode constant_of_shape = FindProducer(producers, fill_value);
    if (!IsOnnxOp(constant_of_shape, "ConstantOfShape")) {
      return false;
    }
    std::vector<Ort::ConstValueInfo> constant_of_shape_inputs =
        constant_of_shape.GetInputs();
    if (constant_of_shape_inputs.size() != 1) {
      return false;
    }
    Ort::ConstNode shape_node =
        FindProducer(producers, constant_of_shape_inputs[0]);
    std::vector<Ort::ConstValueInfo> shape_inputs = shape_node.GetInputs();
    std::vector<Ort::ConstValueInfo> index_clip_outputs =
        index_clip.GetOutputs();
    if (!IsOnnxOp(shape_node, "Shape") || shape_inputs.size() != 1 ||
        index_clip_outputs.size() != 1 ||
        Name(shape_inputs[0]) != Name(index_clip_outputs[0])) {
      return false;
    }
    return AddFusionNode(shape_node, accepted_node_ids, selected_node_ids,
                         fusion_nodes) &&
           AddFusionNode(constant_of_shape, accepted_node_ids,
                         selected_node_ids, fusion_nodes);
  };
  if (!add_fill_shape_chain(low_where_inputs[1]) ||
      !add_fill_shape_chain(high_where_inputs[1])) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> scalar_inputs = {
      reshape_inputs[1],    diff_clip_inputs[1],  time_div_inputs[1],
      value_clip_inputs[1], log_div_inputs[1],    start_sub_inputs[1],
      span_div_inputs[1],   scale_mul_inputs[1],  add_inputs[1],
      index_clip_inputs[1], index_clip_inputs[2], less_inputs[1],
      greater_inputs[1],    unsqueeze_inputs[1]};
  for (Ort::ConstValueInfo input : scalar_inputs) {
    if (!AddConstantProducerChain(input, producers, accepted_node_ids,
                                  selected_node_ids, fusion_nodes)) {
      return false;
    }
  }

  std::unordered_set<size_t> log_bucket_consumer_ids = {
      start_sub.GetId(), less_equal.GetId(), greater_equal.GetId()};
  if (!HasNoExternalConsumers(log_div_outputs[0], graph_output_names,
                              log_bucket_consumer_ids)) {
    return false;
  }

  if (!FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                   selected_node_ids)) {
    return false;
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindLogBucketizeGatherFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  auto producers = BuildProducerMap(all_nodes);
  for (Ort::ConstNode node : all_nodes) {
    if (!IsOnnxOp(node, "Cast") && !IsOnnxOp(node, "Unsqueeze")) {
      continue;
    }
    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseLogBucketizeGather(node, producers, graph_output_names,
                                   accepted_node_ids, fusion_nodes)) {
      continue;
    }
    fusions.push_back(std::move(fusion_nodes));
  }
  return fusions;
}

}  // namespace musa_ep
