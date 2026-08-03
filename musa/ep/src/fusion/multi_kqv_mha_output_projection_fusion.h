#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <memory>
#include <unordered_set>
#include <vector>

#include "fusion/fusion_node_compute.h"

namespace musa_ep {

struct MultiKqvMhaProjectionBranch {
  Ort::ConstNode slice{nullptr};
  Ort::ConstNode matmul{nullptr};
  Ort::ConstNode split{nullptr};
};

// The source graph uses several independent sequence slices.  Each slice is
// projected to Q/K/V in one MatMul, and the three split outputs are
// concatenated before the Microsoft MHA node.  The output is the inverse
// layout: one output projection per sequence slice followed by a Concat.
struct MultiKqvMhaOutputProjectionPattern {
  std::vector<MultiKqvMhaProjectionBranch> branches;
  Ort::ConstNode q_concat{nullptr};
  Ort::ConstNode k_concat{nullptr};
  Ort::ConstNode v_concat{nullptr};
  Ort::ConstNode q_unsqueeze{nullptr};
  Ort::ConstNode k_unsqueeze{nullptr};
  Ort::ConstNode v_unsqueeze{nullptr};
  Ort::ConstNode mha{nullptr};
  Ort::ConstNode output_gather{nullptr};
  std::vector<Ort::ConstNode> output_slices;
  std::vector<Ort::ConstNode> output_gemms;
  Ort::ConstNode output_concat{nullptr};
};

bool TryBuildMultiKqvMhaOutputProjectionPattern(
    Ort::ConstNode candidate,
    const std::unordered_set<std::string>& graph_output_names,
    MultiKqvMhaOutputProjectionPattern& pattern);

bool IsMultiKqvMhaOutputProjectionFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateMultiKqvMhaOutputProjectionFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);

}  // namespace musa_ep
