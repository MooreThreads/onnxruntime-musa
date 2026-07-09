// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <string>
#include <unordered_set>
#include <vector>

namespace musa_ep {

struct FusionMatch {
  const char* finder;
  bool drop_constant_initializers;
  std::vector<std::vector<Ort::ConstNode>> fusions;
};

std::vector<FusionMatch> FindFusionMatches(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names);

std::vector<std::vector<Ort::ConstNode>> FindSplitUnsqueezeConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindSplitConcatReorderFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindConcatMatMulFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindConcatSplitFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindSliceConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindTileConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindGemmActivationFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindFusedGemmFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindShapeReshapeFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindCenteredReduceFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindSplitReduceFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindRmsNormFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindModuloGatherFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindParallelMatMulConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindParallelEinsumActivationFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
bool IsParallelEinsumActivationConcatCandidate(
    Ort::ConstNode concat_node,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindMathConcatLogFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindSparseIdToMaskFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindBucketizeGatherFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindMaskedEmbeddingLookupFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindConcatReshapeFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);

}  // namespace musa_ep
