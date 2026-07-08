// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace musa_ep {

bool HasSingleConsumerAt(
    Ort::ConstValueInfo value_info, Ort::ConstNode expected_node,
    int64_t expected_index,
    const std::unordered_set<std::string>& graph_output_names);
bool GetProducer(Ort::ConstValueInfo value_info, Ort::ConstNode& producer);
bool IsSmallIntegerInitializer(Ort::ConstValueInfo input);
std::unordered_map<std::string, Ort::ConstNode> BuildProducerMap(
    const std::vector<Ort::ConstNode>& all_nodes);
Ort::ConstNode FindProducer(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo input);
bool HasOnlyConsumer(Ort::ConstValueInfo output, Ort::ConstNode expected_node,
                     int64_t expected_input_index);
bool AddFusionNode(Ort::ConstNode node,
                   const std::unordered_set<size_t>& fused_node_ids,
                   std::unordered_set<size_t>& selected_node_ids,
                   std::vector<Ort::ConstNode>& fusion_nodes);
bool FusionHasNoExternalPathBetweenSelectedNodes(
    const std::vector<Ort::ConstNode>& fusion_nodes,
    const std::unordered_set<size_t>& selected_node_ids);
bool ReduceAxesInputIsAxis1(Ort::ConstNode reduce_node);
bool IsReduceSumOrProd(Ort::ConstNode node);
bool ReduceAxesAreLastDim(Ort::ConstNode reduce_node, size_t rank);
bool ReduceOutputKeepsLastDim(Ort::ConstValueInfo output,
                              const std::vector<int64_t>& input_shape);
bool ValueHasOnlyConsumers(Ort::ConstValueInfo value_info,
                           Ort::ConstNode expected_consumer);
bool ValueHasExternalConsumerOrGraphOutput(
    Ort::ConstValueInfo value_info, Ort::ConstNode internal_consumer,
    const std::unordered_set<std::string>& graph_output_names);
std::optional<int64_t> ReadScalarIntInitializer(Ort::ConstValueInfo value_info);

}  // namespace musa_ep
