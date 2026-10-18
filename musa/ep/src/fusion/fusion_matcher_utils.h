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
                   const std::unordered_set<size_t>& accepted_node_ids,
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
bool HasCastBoundary(const std::vector<Ort::ConstNode>& fusion_nodes);

// Returns true when `value_name` is produced only from shape arithmetic:
// constants, plus Shape ops over inputs of the current fusion. The output is
// then a pure function of the shapes of the fused inputs listed in
// `source_indices` (empty means the value is constant). Anything that could
// make the value depend on data, on a tensor outside the fusion, or on an op
// that is not shape-only returns false.
bool ValueDependsOnlyOnInputShapes(
    const std::string& value_name,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_map<std::string, size_t>& input_indices,
    const std::unordered_set<std::string>& initializer_names,
    std::vector<size_t>* source_indices);

}  // namespace musa_ep
