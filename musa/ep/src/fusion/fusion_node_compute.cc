// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/fusion_node_compute.h"

#include <exception>
#include <memory>
#include <string>

#include "ep.h"
#include "fusion/centered_reduce_fusion.h"
#include "fusion/concat_matmul_fusion.h"
#include "fusion/concat_split_fusion.h"
#include "fusion/linear_fusion.h"
#include "fusion/shape_reshape_fusion.h"
#include "fusion/slice_concat_fusion.h"
#include "fusion/split_concat_reorder_fusion.h"
#include "fusion/split_reduce_fusion.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "fusion/split_unsqueeze_concat_fusion.h"

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
    Ort::KernelContext ctx(kernel_context);
    RETURN_IF_ERROR(WaitForPendingMusaInputs(ctx));
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
      if (IsCenteredReduceFusionGraph(graph)) {
        ep->GetFusionComputes()[fused_node_name] =
            CreateCenteredReduceFusion(graph, fused_node);
      } else if (IsShapeReshapeFusionGraph(graph)) {
        ep->GetFusionComputes()[fused_node_name] =
            CreateShapeReshapeFusion(graph, fused_node);
      } else if (IsSplitReduceFusionGraph(graph)) {
        ep->GetFusionComputes()[fused_node_name] =
            CreateSplitReduceFusion(graph, fused_node);
      } else if (IsLinearFusionGraph(graph)) {
        ep->GetFusionComputes()[fused_node_name] =
            CreateLinearFusion(graph, fused_node);
      } else if (IsConcatSplitFusionGraph(graph)) {
        ep->GetFusionComputes()[fused_node_name] =
            CreateConcatSplitFusion(graph, fused_node);
      } else if (IsSliceConcatFusionGraph(graph)) {
        ep->GetFusionComputes()[fused_node_name] =
            CreateSliceConcatFusion(graph, fused_node);
      } else if (IsSplitUnsqueezeConcatFusionGraph(graph)) {
        ep->GetFusionComputes()[fused_node_name] =
            CreateSplitUnsqueezeConcatFusion(graph, fused_node);
      } else if (IsSplitConcatReorderFusionGraph(graph)) {
        ep->GetFusionComputes()[fused_node_name] =
            CreateSplitConcatReorderFusion(graph, fused_node);
      } else {
        ep->GetFusionComputes()[fused_node_name] =
            CreateConcatMatMulFusion(graph, fused_node);
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
