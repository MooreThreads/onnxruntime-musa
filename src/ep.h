#pragma once

#include <string>

#include "plugin_ep_utils.h"

class MusaEpFactory;

class MusaEp : public OrtEp {
 public:
  struct Config {
    bool enable_host_fallback = true;
  };

  MusaEp(MusaEpFactory& factory, const Config& config, const OrtLogger& logger);
  ~MusaEp();

  const OrtApi& GetOrtApi() const { return ort_api_; }
  const OrtEpApi& GetEpApi() const { return ep_api_; }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL
  GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                    OrtEpGraphSupportInfo* graph_support_info) noexcept;
  static OrtStatus* ORT_API_CALL GetKernelRegistryImpl(
      OrtEp* this_ptr, const OrtKernelRegistry** kernel_registry) noexcept;

  MusaEpFactory& factory_;
  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  std::string name_;
  Config config_;
  const OrtLogger& logger_;
};
