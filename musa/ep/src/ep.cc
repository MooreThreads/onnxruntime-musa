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

#include "ep.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "ep_factory.h"
#include "ep_profiling.h"
#include "ep_stream.h"
#include "fusion/fusion_matcher.h"
#include "fusion/fusion_node_compute.h"
#include "graph/cpu_preferred_nodes.h"
#include "graph_mermaid_dump.h"
#include "plugin_ep_utils.h"

using namespace musa_ep;

namespace {

std::unordered_set<std::string> GetGraphOutputNames(Ort::ConstGraph graph) {
  std::unordered_set<std::string> graph_output_names;
  for (Ort::ConstValueInfo output : graph.GetOutputs()) {
    graph_output_names.insert(output.GetName());
  }
  return graph_output_names;
}

bool EnvFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool DisableAllFusions() {
  static const bool disabled = EnvFlagEnabled("ORT_MUSA_DISABLE_ALL_FUSIONS");
  return disabled;
}

bool DisableShapeReshapeFusion() {
  static const bool disabled =
      EnvFlagEnabled("ORT_MUSA_DISABLE_SHAPE_RESHAPE_FUSION");
  return disabled;
}

OrtStatus* RegisterFusionMatches(
    const OrtEpApi& ep_api, OrtEpGraphSupportInfo* graph_support_info,
    const std::vector<FusionMatch>& fusion_matches) {
  for (const FusionMatch& fusion_match : fusion_matches) {
    OrtNodeFusionOptions node_fusion_options = {};
    node_fusion_options.ort_version_supported = ORT_API_VERSION;
    node_fusion_options.drop_constant_initializers =
        fusion_match.drop_constant_initializers;
    for (const auto& fusion_nodes : fusion_match.fusions) {
      RETURN_IF_ERROR(ep_api.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }
  }
  return nullptr;
}

std::unordered_set<size_t> GetFusedNodeIds(
    const std::vector<FusionMatch>& fusion_matches) {
  std::unordered_set<size_t> fused_node_ids;
  for (const FusionMatch& fusion_match : fusion_matches) {
    for (const auto& fusion_nodes : fusion_match.fusions) {
      for (Ort::ConstNode node : fusion_nodes) {
        fused_node_ids.insert(node.GetId());
      }
    }
  }
  return fused_node_ids;
}

struct SupportedNodeCandidates {
  std::vector<Ort::ConstNode> nodes;
  std::vector<const OrtNode*> tentative_nodes;
};

const char* DTypeName(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "bfloat16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "bool";
    default:
      return "other";
  }
}

std::string ValueDType(Ort::ConstValueInfo value) {
  if (value == nullptr) return "missing";
  Ort::ConstTypeInfo type_info = value.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) return "non-tensor";
  return DTypeName(type_info.GetTensorTypeAndShapeInfo().GetElementType());
}

std::string FusionNodeIds(const std::vector<Ort::ConstNode>& nodes) {
  std::string ids;
  for (Ort::ConstNode node : nodes) {
    if (!ids.empty()) ids += ',';
    ids += std::to_string(node.GetId());
  }
  return ids.empty() ? "none" : ids;
}

std::string FusionInputDTypes(const std::vector<Ort::ConstNode>& nodes) {
  std::string types;
  for (Ort::ConstNode node : nodes) {
    for (Ort::ConstValueInfo input : node.GetInputs()) {
      if (!types.empty()) types += ',';
      types += ValueDType(input);
    }
  }
  return types.empty() ? "none" : types;
}

std::string FusionOutputDTypes(const std::vector<Ort::ConstNode>& nodes) {
  std::string types;
  for (Ort::ConstNode node : nodes) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      if (!types.empty()) types += ',';
      types += ValueDType(output);
    }
  }
  return types.empty() ? "none" : types;
}

void EmitDTypeDiagnostics(const std::vector<Ort::ConstNode>& all_nodes,
                          const std::unordered_set<size_t>& registered_node_ids,
                          const std::vector<FusionMatch>& fusion_matches,
                          const MusaProviderOptions& options) {
  if (options.dtype_diagnostics == 0 &&
      options.precision_policy != MusaPrecisionPolicy::Report) {
    return;
  }
  std::fprintf(stderr, "MUSA_DTYPE_REPORT_BEGIN policy=%s nodes=%zu\n",
               MusaPrecisionPolicyName(options.precision_policy),
               all_nodes.size());
  for (const auto& node : all_nodes) {
    const std::string domain =
        node.GetDomain().empty() ? "ai.onnx" : node.GetDomain();
    std::string input_types;
    for (Ort::ConstValueInfo input : node.GetInputs()) {
      if (!input_types.empty()) input_types += ',';
      input_types += ValueDType(input);
    }
    std::fprintf(stderr,
                 "MUSA_DTYPE_NODE id=%zu name=%s op=%s domain=%s inputs=%s "
                 "assigned=%d\n",
                 node.GetId(), node.GetName().c_str(),
                 node.GetOperatorType().c_str(), domain.c_str(),
                 input_types.c_str(),
                 registered_node_ids.count(node.GetId()) != 0 ? 1 : 0);
  }
  for (const FusionMatch& match : fusion_matches) {
    const FusionDTypeContract& contract = match.dtype_contract;
    for (const auto& fusion_nodes : match.fusions) {
      std::fprintf(stderr,
                   "MUSA_DTYPE_FUSION finder=%s nodes=%s inputs=%s outputs=%s "
                   "storage=%s accumulator=%s output_policy=\"%s\" "
                   "cast_policy=\"%s\" accepted=1\n",
                   match.finder, FusionNodeIds(fusion_nodes).c_str(),
                   FusionInputDTypes(fusion_nodes).c_str(),
                   FusionOutputDTypes(fusion_nodes).c_str(),
                   contract.storage_types, contract.accumulator_type,
                   contract.output_type_policy, contract.cast_policy);
    }
    for (const FusionDTypeRejection& rejection : match.dtype_rejections) {
      std::fprintf(stderr,
                   "MUSA_DTYPE_FUSION finder=%s nodes=%s inputs=%s outputs=%s "
                   "storage=%s accumulator=%s output_policy=\"%s\" "
                   "cast_policy=\"%s\" accepted=0 reason=\"%s\"\n",
                   match.finder, FusionNodeIds(rejection.nodes).c_str(),
                   FusionInputDTypes(rejection.nodes).c_str(),
                   FusionOutputDTypes(rejection.nodes).c_str(),
                   contract.storage_types, contract.accumulator_type,
                   contract.output_type_policy, contract.cast_policy,
                   rejection.reason.c_str());
    }
  }
  std::fprintf(stderr, "MUSA_DTYPE_REPORT_END\n");
  std::fflush(stderr);
}

OrtStatus* CollectSupportedNodeCandidates(
    const OrtEpApi& ep_api, const std::string& ep_name,
    OrtEpGraphSupportInfo* graph_support_info,
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<size_t>& fused_node_ids,
    SupportedNodeCandidates& candidates) {
  // Mark non-fused nodes as supported if we have a registered kernel. Defer
  // adding them until after the CUDA-style CPU-preferred shape subgraph pass.
  for (const auto& node : all_nodes) {
    if (fused_node_ids.count(node.GetId()) != 0) {
      continue;
    }

    std::string node_ep_name = node.GetEpName();
    if (!node_ep_name.empty()) {
      if (node_ep_name == ep_name) {
        candidates.nodes.push_back(node);
        candidates.tentative_nodes.push_back(node);
      }
      continue;
    }

    const OrtKernelDef* kernel_def = nullptr;
    RETURN_IF_ERROR(ep_api.EpGraphSupportInfo_LookUpKernel(graph_support_info,
                                                           node, &kernel_def));
    if (kernel_def != nullptr) {
      candidates.nodes.push_back(node);
      candidates.tentative_nodes.push_back(node);
    }
  }
  return nullptr;
}

OrtStatus* RegisterSingleNodeCapabilities(
    const OrtEpApi& ep_api, const OrtGraph& ort_graph,
    OrtEpGraphSupportInfo& graph_support_info,
    const SupportedNodeCandidates& candidates,
    std::unordered_set<size_t>& registered_node_ids) {
  std::unordered_set<const OrtNode*> cpu_preferred_nodes;
  if (!candidates.tentative_nodes.empty()) {
    RETURN_IF_ERROR(GetCpuPreferredNodes(
        ort_graph, graph_support_info, ep_api,
        std::span<const OrtNode* const>(candidates.tentative_nodes.data(),
                                        candidates.tentative_nodes.size()),
        cpu_preferred_nodes));
  }

  for (const auto& node : candidates.nodes) {
    if (cpu_preferred_nodes.count(node) == 0) {
      RETURN_IF_ERROR(
          ep_api.EpGraphSupportInfo_AddSingleNode(&graph_support_info, node));
      registered_node_ids.insert(node.GetId());
    }
  }
  return nullptr;
}

}  // namespace

MusaEp::MusaEp(MusaEpFactory& factory, const Config& config,
               const OrtLogger& logger)
    : OrtEp{},  // explicitly call the struct ctor to ensure all optional values
                // are default initialized
      factory_{factory},
      ort_api_{factory.GetOrtApi()},
      ep_api_{factory.GetEpApi()},
      name_{factory.GetEpName()},
      config_{config},
      logger_{logger} {
  ort_version_supported =
      ORT_API_VERSION;  // set to the ORT version we were compiled with.

  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  CreateProfiler = CreateProfilerImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

  IGNORE_ORTSTATUS(ort_api_.Logger_LogMessage(
      &logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
      ("MUSAExecutionProvider has been created with name " + name_).c_str(),
      ORT_FILE, __LINE__, __FUNCTION__));
}

MusaEp::~MusaEp() = default;

/*static*/
OrtStatus* ORT_API_CALL MusaEp::CreateSyncStreamForDeviceImpl(
    OrtEp* this_ptr, const OrtMemoryDevice* memory_device,
    OrtSyncStreamImpl** stream) noexcept {
  auto& ep = *static_cast<MusaEp*>(this_ptr);
  *stream = nullptr;

  if (ep.ep_api_.MemoryDevice_GetDeviceType(memory_device) !=
          OrtMemoryInfoDeviceType_GPU ||
      ep.ep_api_.MemoryDevice_GetVendorId(memory_device) !=
          ep.factory_.VendorId() ||
      ep.ep_api_.MemoryDevice_GetMemoryType(memory_device) !=
          OrtDeviceMemoryType_DEFAULT) {
    return nullptr;
  }

  const MusaProviderOptions& options = ep.config_.provider_options;

  try {
    if (options.has_user_compute_stream != 0) {
      auto sync_stream = std::make_unique<MusaSyncStream>(
          ep.ort_api_, options.user_compute_stream);
      *stream = sync_stream.release();
      return nullptr;
    }

    auto sync_stream = std::make_unique<MusaSyncStream>(ep.ort_api_);
    *stream = sync_stream.release();
    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return ep.ort_api_.CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

/*static*/
const char* ORT_API_CALL MusaEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const MusaEp*>(this_ptr);
  return ep->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL
MusaEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                          OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    MusaEp* ep = static_cast<MusaEp*>(this_ptr);

    RETURN_IF_ERROR(DumpGetCapabilityGraphToMermaidIfEnabled(*ort_graph));

    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> all_nodes = graph.GetNodes();

    if (all_nodes.empty()) {
      return nullptr;  // No nodes to process
    }

    std::unordered_set<std::string> graph_output_names =
        GetGraphOutputNames(graph);
    std::vector<FusionMatch> fusion_matches;
    if (!DisableAllFusions()) {
      fusion_matches = FindFusionMatches(all_nodes, graph_output_names);
      if (DisableShapeReshapeFusion()) {
        fusion_matches.erase(
            std::remove_if(fusion_matches.begin(), fusion_matches.end(),
                           [](const FusionMatch& match) {
                             return std::string(match.finder) ==
                                    "FindShapeReshapeFusions";
                           }),
            fusion_matches.end());
      }
      RETURN_IF_ERROR(RegisterFusionMatches(ep->ep_api_, graph_support_info,
                                            fusion_matches));
    }

    SupportedNodeCandidates candidates;
    RETURN_IF_ERROR(CollectSupportedNodeCandidates(
        ep->ep_api_, ep->name_, graph_support_info, all_nodes,
        GetFusedNodeIds(fusion_matches), candidates));
    std::unordered_set<size_t> registered_node_ids =
        GetFusedNodeIds(fusion_matches);
    RETURN_IF_ERROR(RegisterSingleNodeCapabilities(
        ep->ep_api_, *ort_graph, *graph_support_info, candidates,
        registered_node_ids));
    EmitDTypeDiagnostics(all_nodes, registered_node_ids, fusion_matches,
                         ep->config_.provider_options);
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL MusaEp::GetKernelRegistryImpl(
    _In_ OrtEp* this_ptr, _Outptr_result_maybenull_ const OrtKernelRegistry**
                              kernel_registry) noexcept {
  MusaEp* ep = static_cast<MusaEp*>(this_ptr);

  *kernel_registry = nullptr;

  // Get the cached kernel registry from parent factory to avoid recreating the
  // kernel registry for every EP instance.
  RETURN_IF_ERROR(ep->factory_.GetKernelRegistryForEp(*ep, kernel_registry));
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL MusaEp::CreateProfilerImpl(
    OrtEp* this_ptr, OrtEpProfilerImpl** profiler) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  MusaEp* ep = static_cast<MusaEp*>(this_ptr);
  auto profiler_unique_ptr = std::make_unique<MusaEpProfiler>(ep->ep_api_);

  *profiler = profiler_unique_ptr.release();
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}
