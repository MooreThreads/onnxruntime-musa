#pragma once

#include <memory>
#include <string>

#include "plugin_ep_utils.h"

class MusaEp;

class MusaEpFactory : public OrtEpFactory {
 public:
  MusaEpFactory(const OrtApi& ort_api, const OrtEpApi& ep_api,
                const OrtLogger& default_logger);
  ~MusaEpFactory();

  const OrtApi& GetOrtApi() const { return ort_api_; }
  const OrtEpApi& GetEpApi() const { return ep_api_; }
  const std::string& GetEpName() const { return ep_name_; }

  OrtStatus* GetKernelRegistryForEp(MusaEp& ep,
                                    const OrtKernelRegistry** kernel_registry);

 private:
  static const char* ORT_API_CALL
  GetNameImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL
  GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL
  GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL
  GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(
      OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
      size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
      size_t* num_ep_devices) noexcept;
  static OrtStatus* ORT_API_CALL
  CreateEpImpl(OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
               const OrtKeyValuePairs* const* ep_metadata, size_t num_devices,
               const OrtSessionOptions* session_options,
               const OrtLogger* logger, OrtEp** ep) noexcept;
  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* this_ptr,
                                         OrtEp* ep) noexcept;
  static OrtStatus* ORT_API_CALL
  CreateAllocatorImpl(OrtEpFactory* this_ptr, const OrtMemoryInfo* memory_info,
                      const OrtKeyValuePairs* allocator_options,
                      OrtAllocator** allocator) noexcept;
  static void ORT_API_CALL ReleaseAllocatorImpl(
      OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept;

  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  const std::string ep_name_{"MusaExecutionProvider"};
  const std::string vendor_{"MooreThreads"};
  const uint32_t vendor_id_{0x1ED5};
  const std::string ep_version_{"0.1.0"};
  Ort::MemoryInfo default_memory_info_{nullptr};
  Ort::MemoryInfo readonly_memory_info_{nullptr};
  OrtKernelRegistry* kernel_registry_ = nullptr;
};
