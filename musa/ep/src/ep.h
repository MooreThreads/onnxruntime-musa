// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <memory>
#include <string>
#include <unordered_map>

#include "ep_options.h"

class FusionNodeCompute;
class MusaEpFactory;

/// <summary>
/// Example EP that uses kernel registration.
/// </summary>
class MusaEp : public OrtEp {
 public:
  struct Config {
    bool enable_prepack_weight_sharing = false;
    MusaProviderOptions provider_options;
  };

  MusaEp(MusaEpFactory& factory, const Config& config, const OrtLogger& logger);
  ~MusaEp();

  const OrtApi& GetOrtApi() const { return ort_api_; }
  const OrtEpApi& GetEpApi() const { return ep_api_; }
  const Config& GetConfig() const { return config_; }
  std::unordered_map<std::string, std::unique_ptr<FusionNodeCompute>>&
  GetFusionComputes() {
    return fusion_computes_;
  }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetKernelRegistryImpl(
      _In_ OrtEp* this_ptr, _Outptr_result_maybenull_ const OrtKernelRegistry**
                                kernel_registry) noexcept;

  static OrtStatus* ORT_API_CALL
  GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                    OrtEpGraphSupportInfo* graph_support_info) noexcept;

  static OrtStatus* ORT_API_CALL
  CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** graphs,
              _In_ const OrtNode** fused_nodes, _In_ size_t count,
              _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
              _Out_writes_(count) OrtNode** ep_context_nodes) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
      OrtEp* this_ptr, OrtNodeComputeInfo** node_compute_infos,
      size_t num_node_compute_infos) noexcept;

  static OrtStatus* ORT_API_CALL
  CreateProfilerImpl(OrtEp* this_ptr, OrtEpProfilerImpl** profiler) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEp* this_ptr, const OrtMemoryDevice* memory_device,
      OrtSyncStreamImpl** stream) noexcept;

  MusaEpFactory& factory_;
  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  std::string name_;
  Config config_;
  const OrtLogger& logger_;
  std::unordered_map<std::string, std::unique_ptr<FusionNodeCompute>>
      fusion_computes_;
};
