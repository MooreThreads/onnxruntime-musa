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

#include "fusion/fusion_node_compute.h"

#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <vector>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

#include "ep.h"
#include "fusion/bucketize_gather_fusion.h"
#include "fusion/centered_reduce_fusion.h"
#include "fusion/concat_matmul_fusion.h"
#include "fusion/concat_reshape_fusion.h"
#include "fusion/concat_split_fusion.h"
#include "fusion/linear_fusion.h"
#include "fusion/masked_embedding_lookup_fusion.h"
#include "fusion/math_concat_log_fusion.h"
#include "fusion/mhta_scaled_dot_product_attention_fusion.h"
#include "fusion/modulo_gather_fusion.h"
#include "fusion/parallel_einsum_activation_fusion.h"
#include "fusion/parallel_linear_fusion.h"
#include "fusion/parallel_matmul_concat_fusion.h"
#include "fusion/replace_invalid_id_fusion.h"
#include "fusion/rms_norm_fusion.h"
#include "fusion/segment_max_broadcast_fusion.h"
#include "fusion/shape_reshape_fusion.h"
#include "fusion/slice_concat_fusion.h"
#include "fusion/sparse_id_to_mask_fusion.h"
#include "fusion/split_concat_reorder_fusion.h"
#include "fusion/split_reduce_fusion.h"
#include "fusion/split_unsqueeze_concat_fusion.h"
#include "fusion/target_id_count_embedding_fusion.h"
#include "fusion/tile_concat_fusion.h"
#include "plugin_ep_utils.h"
#include "runtime_graph_dump.h"

/*
 * Fusion node runtime bridge
 *
 * This file does not implement a math operator. It connects ORT Plugin EP
 * fused nodes to the concrete FusionNodeCompute objects created for each
 * supported graph pattern.
 *
 * Flow:
 *   1. MusaEp::GetCapability finds a supported subgraph pattern and asks ORT
 *      to replace it with a fused node assigned to the MUSA EP.
 *   2. MusaEp::CompileImpl receives that fused node and stores one concrete
 *      FusionNodeCompute object in MusaEp::fusion_computes_, keyed by fused
 *      node name.
 *   3. At runtime ORT calls OrtNodeComputeInfo::Compute for the fused node.
 *      FusionNodeComputeInfo looks up the stored FusionNodeCompute by node name
 *      and forwards the OrtKernelContext to the pattern-specific Compute().
 */

namespace {

std::vector<std::string> ValueInfoNames(
    const std::vector<Ort::ConstValueInfo>& value_infos) {
  std::vector<std::string> names;
  names.reserve(value_infos.size());
  for (Ort::ConstValueInfo value_info : value_infos) {
    if (value_info != nullptr) {
      names.push_back(value_info.GetName());
    }
  }
  return names;
}

RuntimeGraphNodeMetadata CreateFusionRuntimeMetadata(
    const Ort::ConstGraph& graph, const Ort::ConstNode& fused_node,
    const std::string& display_type) {
  RuntimeGraphNodeMetadata metadata;
  metadata.kind = "fusion";
  metadata.display_type = display_type;
  metadata.node_name = fused_node.GetName();
  metadata.domain_name = fused_node.GetDomain();
  metadata.since_version = fused_node.GetSinceVersion();
  metadata.inputs = ValueInfoNames(fused_node.GetInputs());
  metadata.outputs = ValueInfoNames(fused_node.GetOutputs());

  std::unordered_set<std::string> seen_source_ops;
  for (Ort::ConstNode source_node : graph.GetNodes()) {
    std::string source_name = source_node.GetName();
    if (source_name.empty()) {
      source_name = source_node.GetOperatorType();
    }
    metadata.source_nodes.push_back(std::move(source_name));

    std::string source_op = source_node.GetOperatorType();
    if (seen_source_ops.insert(source_op).second) {
      metadata.source_ops.push_back(std::move(source_op));
    }
  }
  return metadata;
}

std::string RuntimeTypeName(const FusionNodeCompute& compute) {
  std::string name;
#if defined(__GNUG__)
  int status = 0;
  char* demangled =
      abi::__cxa_demangle(typeid(compute).name(), nullptr, nullptr, &status);
  if (status == 0 && demangled != nullptr) {
    name = demangled;
  }
  std::free(demangled);
#endif
  if (name.empty()) {
    name = typeid(compute).name();
  }

  const size_t namespace_pos = name.rfind("::");
  if (namespace_pos != std::string::npos) {
    name = name.substr(namespace_pos + 2);
  }
  return name;
}

struct FusionNodeComputeInfo : OrtNodeComputeInfo {
  explicit FusionNodeComputeInfo(MusaEp& ep) : ep(ep) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  static OrtStatus* ORT_API_CALL CreateStateImpl(
      OrtNodeComputeInfo* this_ptr, OrtNodeComputeContext* compute_context,
      void** compute_state) {
    auto* self = static_cast<FusionNodeComputeInfo*>(this_ptr);
    MusaEp& ep = self->ep;
    std::string fused_node_name =
        ep.GetEpApi().NodeComputeContext_NodeName(compute_context);

    auto it = ep.GetFusionComputes().find(fused_node_name);
    if (it == ep.GetFusionComputes().end()) {
      std::string message =
          "Unable to get MUSA fusion compute for fused node " + fused_node_name;
      return ep.GetOrtApi().CreateStatus(ORT_EP_FAIL, message.c_str());
    }

    *compute_state = it->second.get();
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                             void* compute_state,
                                             OrtKernelContext* kernel_context) {
    auto* fusion = reinterpret_cast<const FusionNodeCompute*>(compute_state);
    if (RuntimeGraphDumpEnabled()) {
      struct RuntimeComputeScope {
        explicit RuntimeComputeScope(const void* impl,
                                     const char* fallback_label)
            : id(BeginRuntimeCompute(impl, fallback_label)) {}
        ~RuntimeComputeScope() { EndRuntimeCompute(id); }
        uint64_t id;
      } runtime_compute_scope(fusion, "FusionNodeCompute");
      return fusion->Compute(kernel_context);
    }
    return fusion->Compute(kernel_context);
  }

  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                            void* /*compute_state*/) {}

  MusaEp& ep;
};

}  // namespace

OrtNodeComputeInfo* CreateFusionNodeComputeInfo(MusaEp& ep) {
  return new FusionNodeComputeInfo(ep);
}

void ReleaseFusionNodeComputeInfo(OrtNodeComputeInfo* node_compute_info) {
  delete static_cast<FusionNodeComputeInfo*>(node_compute_info);
}

OrtStatus* ORT_API_CALL MusaEp::CompileImpl(
    _In_ OrtEp* this_ptr, _In_ const OrtGraph** graphs,
    _In_ const OrtNode** fused_nodes, _In_ size_t count,
    _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
    _Out_writes_(count) OrtNode** ep_context_nodes) noexcept {
  try {
    auto* ep = static_cast<MusaEp*>(this_ptr);
    for (size_t i = 0; i < count; ++i) {
      node_compute_infos[i] = nullptr;
      if (ep_context_nodes != nullptr) {
        ep_context_nodes[i] = nullptr;
      }

      Ort::ConstGraph graph{graphs[i]};
      Ort::ConstNode fused_node{fused_nodes[i]};
      if (fused_node.GetEpName() != ep->name_) {
        return ep->ort_api_.CreateStatus(
            ORT_EP_FAIL, "MUSA fused node is not assigned to MUSA EP");
      }

      std::string fused_node_name = fused_node.GetName();
      auto& fusion_compute = ep->GetFusionComputes()[fused_node_name];
      if (IsMhtaScaledDotProductAttentionFusionGraph(graph)) {
        fusion_compute =
            CreateMhtaScaledDotProductAttentionFusion(graph, fused_node);
      } else if (IsCenteredReduceFusionGraph(graph)) {
        fusion_compute = CreateCenteredReduceFusion(graph, fused_node);
      } else if (IsShapeReshapeFusionGraph(graph)) {
        fusion_compute = CreateShapeReshapeFusion(graph, fused_node);
      } else if (IsSplitReduceFusionGraph(graph)) {
        fusion_compute = CreateSplitReduceFusion(graph, fused_node);
      } else if (IsParallelLinearFusionGraph(graph)) {
        fusion_compute = CreateParallelLinearFusion(graph, fused_node);
      } else if (IsLinearFusionGraph(graph)) {
        fusion_compute = CreateLinearFusion(graph, fused_node);
      } else if (IsConcatSplitFusionGraph(graph)) {
        fusion_compute = CreateConcatSplitFusion(graph, fused_node);
      } else if (IsSliceConcatFusionGraph(graph)) {
        fusion_compute = CreateSliceConcatFusion(graph, fused_node);
      } else if (IsSplitUnsqueezeConcatFusionGraph(graph)) {
        fusion_compute = CreateSplitUnsqueezeConcatFusion(graph, fused_node);
      } else if (IsSplitConcatReorderFusionGraph(graph)) {
        fusion_compute = CreateSplitConcatReorderFusion(graph, fused_node);
      } else if (IsTileConcatFusionGraph(graph)) {
        fusion_compute = CreateTileConcatFusion(graph, fused_node);
      } else if (IsRmsNormFusionGraph(graph)) {
        fusion_compute = CreateRmsNormFusion(graph, fused_node);
      } else if (IsTargetIdCountEmbeddingFusionGraph(graph)) {
        fusion_compute = CreateTargetIdCountEmbeddingFusion(graph, fused_node);
      } else if (IsModuloGatherFusionGraph(graph)) {
        fusion_compute = CreateModuloGatherFusion(graph, fused_node);
      } else if (IsParallelEinsumActivationFusionGraph(graph)) {
        fusion_compute =
            CreateParallelEinsumActivationFusion(graph, fused_node);
      } else if (IsParallelMatMulConcatFusionGraph(graph)) {
        fusion_compute = CreateParallelMatMulConcatFusion(graph, fused_node);
      } else if (IsMathConcatLogFusionGraph(graph)) {
        fusion_compute = CreateMathConcatLogFusion(graph, fused_node);
      } else if (IsSparseIdToMaskFusionGraph(graph)) {
        fusion_compute = CreateSparseIdToMaskFusion(graph, fused_node);
      } else if (IsBucketizeGatherFusionGraph(graph)) {
        fusion_compute = CreateBucketizeGatherFusion(graph, fused_node);
      } else if (IsMaskedEmbeddingLookupFusionGraph(graph)) {
        fusion_compute = CreateMaskedEmbeddingLookupFusion(graph, fused_node);
      } else if (IsConcatReshapeFusionGraph(graph)) {
        fusion_compute = CreateConcatReshapeFusion(graph, fused_node);
      } else if (IsReplaceInvalidIdFusionGraph(graph)) {
        fusion_compute = CreateReplaceInvalidIdFusion(graph, fused_node);
      } else if (IsSegmentMaxBroadcastFusionGraph(graph)) {
        fusion_compute = CreateSegmentMaxBroadcastFusion(graph, fused_node);
      } else {
        fusion_compute = CreateConcatMatMulFusion(graph, fused_node);
      }
      if (RuntimeGraphDumpEnabled()) {
        RegisterRuntimeFusionInstance(
            fusion_compute.get(),
            CreateFusionRuntimeMetadata(graph, fused_node,
                                        RuntimeTypeName(*fusion_compute)));
      }
      node_compute_infos[i] = CreateFusionNodeComputeInfo(*ep);
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

void ORT_API_CALL MusaEp::ReleaseNodeComputeInfosImpl(
    OrtEp* /*this_ptr*/, OrtNodeComputeInfo** node_compute_infos,
    size_t num_node_compute_infos) noexcept {
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    ReleaseFusionNodeComputeInfo(node_compute_infos[i]);
  }
}
