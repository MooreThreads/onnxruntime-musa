// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <span>
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

bool CanFuseSliceConcat(
    Ort::ConstNode concat_node,
    const std::unordered_set<std::string>& graph_output_names,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_inputs.empty() || concat_outputs.size() != 1 ||
      graph_output_names.count(Name(concat_outputs[0])) != 0) {
    return false;
  }

  auto axis_attr = GetIntAttribute(concat_node, "axis");
  int64_t axis = 0;
  if (!axis_attr.has_value() || !NormalizeAxis(*axis_attr, 2, axis) ||
      axis != 1) {
    return false;
  }

  fusion_nodes.clear();
  fusion_nodes.reserve(concat_inputs.size() + 1);
  std::unordered_set<size_t> seen_slice_node_ids;
  bool has_fused_slice_input = false;
  for (Ort::ConstValueInfo concat_input : concat_inputs) {
    Ort::ValueInfoConsumerProducerInfo producer =
        concat_input.GetProducerNode();
    if (!producer.node) {
      auto shape = GetTensorShape(concat_input);
      if (!IsFloatTensorValueInfo(concat_input) || !shape.has_value() ||
          shape->size() != 2 || (*shape)[1] <= 0) {
        return false;
      }
      continue;
    }

    Ort::ConstNode slice_node = producer.node;
    const bool is_slice_input = IsOnnxOp(slice_node, "Slice");
    const bool is_zero_constant_input = IsOnnxOp(slice_node, "ConstantOfShape");
    if (!is_slice_input && !is_zero_constant_input) {
      auto shape = GetTensorShape(concat_input);
      if (!IsFloatTensorValueInfo(concat_input) || !shape.has_value() ||
          shape->size() != 2 || (*shape)[1] <= 0) {
        return false;
      }
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
        concat_input.GetConsumers();
    if (consumers.empty()) {
      return false;
    }
    for (const auto& consumer : consumers) {
      if (consumer.node.GetId() != concat_node.GetId()) {
        return false;
      }
    }

    has_fused_slice_input = true;
    if (IsOnnxOp(slice_node, "ConstantOfShape")) {
      if (!IsZeroFloatConstantOfShape(slice_node)) {
        return false;
      }
      auto output_shape = ConstantOfShapeOutputShape(slice_node, concat_input);
      if (!output_shape.has_value()) {
        return false;
      }
      std::vector<Ort::ConstValueInfo> constant_inputs = slice_node.GetInputs();
      if (constant_inputs.size() == 1) {
        Ort::ValueInfoConsumerProducerInfo shape_producer =
            constant_inputs[0].GetProducerNode();
        if (shape_producer.node && IsOnnxOp(shape_producer.node, "Shape")) {
          std::vector<Ort::ValueInfoConsumerProducerInfo> shape_consumers =
              constant_inputs[0].GetConsumers();
          if (shape_consumers.size() != 1 ||
              shape_consumers[0].node.GetId() != slice_node.GetId()) {
            return false;
          }
          if (seen_slice_node_ids.insert(shape_producer.node.GetId()).second) {
            fusion_nodes.push_back(shape_producer.node);
          }
        }
      }
      if (seen_slice_node_ids.insert(slice_node.GetId()).second) {
        fusion_nodes.push_back(slice_node);
      }
      continue;
    }

    std::vector<Ort::ConstValueInfo> slice_inputs = slice_node.GetInputs();
    if (slice_inputs.size() < 3 || slice_inputs.size() > 5) {
      return false;
    }

    auto input_shape = GetTensorShape(slice_inputs[0]);
    if (input_shape.has_value() && input_shape->size() != 2) {
      return false;
    }

    auto starts = ReadIntInitializerNoLimit(slice_inputs[1]);
    auto ends = ReadIntInitializerNoLimit(slice_inputs[2]);
    if (!starts.has_value() || !ends.has_value() ||
        starts->size() != ends->size()) {
      return false;
    }
    std::vector<int64_t> axes(starts->size());
    std::iota(axes.begin(), axes.end(), 0);
    if (slice_inputs.size() > 3 && slice_inputs[3]) {
      auto axes_init = ReadIntInitializerNoLimit(slice_inputs[3]);
      if (!axes_init.has_value()) {
        return false;
      }
      axes = *axes_init;
    }
    std::vector<int64_t> steps(starts->size(), 1);
    if (slice_inputs.size() > 4 && slice_inputs[4]) {
      auto steps_init = ReadIntInitializerNoLimit(slice_inputs[4]);
      if (!steps_init.has_value()) {
        return false;
      }
      steps = *steps_init;
    }
    if (axes.size() != starts->size() || steps.size() != starts->size()) {
      return false;
    }

    bool has_col_slice = false;
    for (size_t i = 0; i < axes.size(); ++i) {
      int64_t slice_axis = axes[i] < 0 ? axes[i] + 2 : axes[i];
      if (slice_axis < 0 || slice_axis > 1 || steps[i] != 1) {
        return false;
      }
      if (slice_axis == 0) {
        if ((*starts)[i] != 0 ||
            (*ends)[i] < std::numeric_limits<int64_t>::max() / 4) {
          return false;
        }
      } else {
        if (((*starts)[i] < 0 || (*ends)[i] < 0) &&
            (!input_shape.has_value() || (*input_shape)[1] <= 0)) {
          return false;
        }
        int64_t start =
            (*starts)[i] < 0 ? (*starts)[i] + (*input_shape)[1] : (*starts)[i];
        int64_t end =
            (*ends)[i] < 0 ? (*ends)[i] + (*input_shape)[1] : (*ends)[i];
        if (input_shape.has_value() && (*input_shape)[1] > 0) {
          start = std::max<int64_t>(0, std::min(start, (*input_shape)[1]));
          end = std::max<int64_t>(0, std::min(end, (*input_shape)[1]));
        }
        if (end <= start) {
          return false;
        }
        has_col_slice = true;
      }
    }
    if (!has_col_slice) {
      return false;
    }

    if (seen_slice_node_ids.insert(slice_node.GetId()).second) {
      fusion_nodes.push_back(slice_node);
    }
  }

  fusion_nodes.push_back(concat_node);
  return has_fused_slice_input;
}

std::vector<std::vector<Ort::ConstNode>> FindSliceConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode concat_node : all_nodes) {
    if (!IsOnnxOp(concat_node, "Concat") ||
        accepted_node_ids.count(concat_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstNode> fusion_nodes;
    if (!CanFuseSliceConcat(concat_node, graph_output_names, fusion_nodes)) {
      continue;
    }

    bool overlaps_existing_fusion = false;
    for (Ort::ConstNode node : fusion_nodes) {
      if (accepted_node_ids.count(node.GetId()) != 0) {
        overlaps_existing_fusion = true;
        break;
      }
    }
    if (overlaps_existing_fusion) {
      continue;
    }
    fusions.push_back(std::move(fusion_nodes));
  }

  return fusions;
}

}  // namespace musa_ep
