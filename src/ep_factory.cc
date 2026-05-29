#include "ep_factory.h"

#include <memory>

#include "allocator.h"
#include "ep.h"
#include "ep_kernel_registration.h"
#include "musa/musa_backend.h"

MusaEpFactory::MusaEpFactory(const OrtApi& ort_api, const OrtEpApi& ep_api,
                             const OrtLogger& default_logger)
    : OrtEpFactory{}, ort_api_(ort_api), ep_api_(ep_api) {
  ort_version_supported = ORT_API_VERSION;
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;

  default_memory_info_ = Ort::MemoryInfo{"MusaExecutionProvider host",
                                         OrtMemoryInfoDeviceType_CPU,
                                         vendor_id_,
                                         0,
                                         OrtDeviceMemoryType_DEFAULT,
                                         0,
                                         OrtAllocatorType::OrtDeviceAllocator};
  readonly_memory_info_ =
      Ort::MemoryInfo{"MusaExecutionProvider host readonly",
                      OrtMemoryInfoDeviceType_CPU,
                      vendor_id_,
                      0,
                      OrtDeviceMemoryType_DEFAULT,
                      0,
                      OrtAllocatorType::OrtReadOnlyAllocator};

  const auto runtime_info = ort_musa::QueryMusaRuntime();
  IGNORE_ORT_STATUS(ort_api_.Logger_LogMessage(
      &default_logger, ORT_LOGGING_LEVEL_INFO, runtime_info.description.c_str(),
      ORT_MUSA_FILE, __LINE__, __FUNCTION__));
}

MusaEpFactory::~MusaEpFactory() {
  if (kernel_registry_ != nullptr) {
    Ort::GetEpApi().ReleaseKernelRegistry(kernel_registry_);
  }
}

OrtStatus* MusaEpFactory::GetKernelRegistryForEp(
    MusaEp& ep, const OrtKernelRegistry** out_kernel_registry) {
  *out_kernel_registry = nullptr;
  if (GetNumKernels() == 0) {
    return nullptr;
  }
  if (kernel_registry_ == nullptr) {
    RETURN_IF_ERROR(
        CreateKernelRegistry(ep.GetName(static_cast<const OrtEp*>(&ep)),
                             nullptr, &kernel_registry_));
  }
  *out_kernel_registry = kernel_registry_;
  return nullptr;
}

const char* ORT_API_CALL
MusaEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const MusaEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

const char* ORT_API_CALL
MusaEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const MusaEpFactory*>(this_ptr);
  return factory->vendor_.c_str();
}

uint32_t ORT_API_CALL
MusaEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const MusaEpFactory*>(this_ptr);
  return factory->vendor_id_;
}

const char* ORT_API_CALL
MusaEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const MusaEpFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

OrtStatus* ORT_API_CALL MusaEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
    size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  auto* factory = static_cast<MusaEpFactory*>(this_ptr);
  *p_num_ep_devices = 0;

  for (size_t i = 0; i < num_devices && *p_num_ep_devices < max_ep_devices;
       ++i) {
    const OrtHardwareDevice& device = *devices[i];
    if (factory->ort_api_.HardwareDevice_Type(&device) !=
        OrtHardwareDeviceType_CPU) {
      continue;
    }

    OrtKeyValuePairs* metadata = nullptr;
    OrtKeyValuePairs* options = nullptr;
    factory->ort_api_.CreateKeyValuePairs(&metadata);
    factory->ort_api_.CreateKeyValuePairs(&options);

    const auto runtime_info = ort_musa::QueryMusaRuntime();
    factory->ort_api_.AddKeyValuePair(metadata, "backend", "musa");
    factory->ort_api_.AddKeyValuePair(metadata, "musa_runtime",
                                      runtime_info.description.c_str());
    factory->ort_api_.AddKeyValuePair(options, "host_fallback", "1");

    OrtEpDevice* ep_device = nullptr;
    OrtStatus* status = factory->ep_api_.CreateEpDevice(
        factory, &device, metadata, options, &ep_device);
    factory->ort_api_.ReleaseKeyValuePairs(metadata);
    factory->ort_api_.ReleaseKeyValuePairs(options);
    RETURN_IF_ERROR(status);

    RETURN_IF_ERROR(factory->ep_api_.EpDevice_AddAllocatorInfo(
        ep_device, factory->default_memory_info_));
    RETURN_IF_ERROR(factory->ep_api_.EpDevice_AddAllocatorInfo(
        ep_device, factory->readonly_memory_info_));
    ep_devices[(*p_num_ep_devices)++] = ep_device;
  }

  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL MusaEpFactory::CreateEpImpl(
    OrtEpFactory* this_ptr, const OrtHardwareDevice* const*,
    const OrtKeyValuePairs* const*, size_t num_devices,
    const OrtSessionOptions* session_options, const OrtLogger* logger,
    OrtEp** ep) noexcept {
  auto* factory = static_cast<MusaEpFactory*>(this_ptr);
  *ep = nullptr;
  if (num_devices != 1) {
    return factory->ort_api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "MusaExecutionProvider expects one selected device.");
  }

  std::string host_fallback;
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(
      *session_options, "ep.musa.enable_host_fallback", "1", host_fallback));
  MusaEp::Config config;
  config.enable_host_fallback = host_fallback != "0";

  auto instance = std::make_unique<MusaEp>(*factory, config, *logger);
  *ep = instance.release();
  return nullptr;
}

void ORT_API_CALL MusaEpFactory::ReleaseEpImpl(OrtEpFactory*,
                                               OrtEp* ep) noexcept {
  delete static_cast<MusaEp*>(ep);
}

OrtStatus* ORT_API_CALL MusaEpFactory::CreateAllocatorImpl(
    OrtEpFactory* this_ptr, const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs*, OrtAllocator** allocator) noexcept {
  auto* factory = static_cast<MusaEpFactory*>(this_ptr);
  *allocator = nullptr;
  if (memory_info != factory->default_memory_info_ &&
      memory_info != factory->readonly_memory_info_) {
    return factory->ort_api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "Unknown memory info for MusaExecutionProvider allocator.");
  }
  *allocator = new MusaHostAllocator(memory_info);
  return nullptr;
}

void ORT_API_CALL MusaEpFactory::ReleaseAllocatorImpl(
    OrtEpFactory*, OrtAllocator* allocator) noexcept {
  delete static_cast<MusaHostAllocator*>(allocator);
}
