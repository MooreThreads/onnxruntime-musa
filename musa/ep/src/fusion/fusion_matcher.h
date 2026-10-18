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

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fusion/fusion_dtype.h"
#include "fusion/shape_expr.h"

namespace musa_ep {

// Shape expressions a finder proved on the whole graph, keyed first by a
// signature of the fusion boundary and then by the outer value name they
// compute. GetCapability() fills this in; Compile() reads it back because the
// subgraph it receives does not contain the nodes the value is computed from.
using FusionShapeExprHints =
    std::unordered_map<std::string,
                       std::unordered_map<std::string, ShapeExprHint>>;

// Mask-end provenance a finder proved on the whole graph, keyed the same way:
// boundary signature -> end value name -> names of the fused inputs whose
// shapes determine that value. Compile() cannot see the end's producer chain
// and remaps these names to fused input indices.
using FusionMaskEndHints =
    std::unordered_map<std::string,
                       std::unordered_map<std::string, std::vector<std::string>>>;

struct FusionDTypeRejection {
  std::vector<Ort::ConstNode> nodes;
  std::string reason;
};

struct FusionMatch {
  const char* finder;
  bool drop_constant_initializers;
  FusionDTypeContract dtype_contract;
  std::vector<std::vector<Ort::ConstNode>> fusions;
  std::vector<FusionDTypeRejection> dtype_rejections;
};

std::vector<FusionMatch> FindFusionMatches(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    FusionShapeExprHints* shape_expr_hints = nullptr,
    FusionMaskEndHints* mask_end_hints = nullptr);

std::vector<std::vector<Ort::ConstNode>> FindSplitUnsqueezeConcatFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindSplitConcatFusions(
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
    const std::unordered_set<size_t>& accepted_node_ids,
    FusionShapeExprHints* shape_expr_hints = nullptr);
std::vector<std::vector<Ort::ConstNode>> FindGemmActivationFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindParallelLinearFusions(
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
std::vector<std::vector<Ort::ConstNode>> FindCastRmsNormFusions(
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
std::vector<std::vector<Ort::ConstNode>> FindSwiGluFusions(
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
std::vector<std::vector<Ort::ConstNode>> FindLogBucketizeGatherFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindMaskedEmbeddingLookupFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindTargetIdCountEmbeddingFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindConcatReshapeFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindReplaceInvalidIdFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindSegmentMaxBroadcastFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindRecRankCalibrationFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindStridedViewFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>>
FindMhtaScaledDotProductAttentionFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    FusionMaskEndHints* mask_end_hints = nullptr);
std::vector<std::vector<Ort::ConstNode>>
FindQkvAttentionOutputProjectionFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindMultiKqvMhaOutputProjectionFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindMoEFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindGenerateMTGRCustomMaskFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindSplitSequenceMoEFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
std::vector<std::vector<Ort::ConstNode>> FindSiluFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);

}  // namespace musa_ep
