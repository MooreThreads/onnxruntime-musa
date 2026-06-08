// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_factory.h"

#include <musa_runtime.h>

#include <cassert>

#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "ep.h"
#include "ep_allocator.h"
#include "ep_kernel_registration.h"
#include "ep_stream.h"
#include "plugin_ep_utils.h"

MusaEpFactory::MusaEpFactory(const OrtApi& ort_api, const OrtEpApi& ep_api,
                             const OrtLogger& /*default_logger*/)
    : OrtEpFactory{},
      ort_api_(ort_api),
      ep_api_(ep_api),
      default_memory_info_{nullptr},
      readonly_memory_info_{nullptr} {
  ort_version_supported =
      ORT_API_VERSION;  // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;

  GetSupportedDevices = GetSupportedDevicesImpl;

  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;

  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;

  CreateDataTransfer = CreateDataTransferImpl;

  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

  // Define the default memory info. Allows creating custom OrtAllocators and
  // OrtDataTransferImpls. This is not strictly required for cpu-based EPs, like
  // this example EP. However, we define it here to serve as an example for
  // non-cpu EPs.
  default_memory_info_ = Ort::MemoryInfo{"MUSAExecutionProvider CPU",
                                         OrtMemoryInfoDeviceType_GPU,
                                         vendor_id_,
                                         /* device_id */ 0,
                                         OrtDeviceMemoryType_DEFAULT,
                                         /*alignment*/ 0,
                                         OrtAllocatorType::OrtDeviceAllocator};

  // create data transfer for the device
  const OrtMemoryDevice* device =
      ep_api.MemoryInfo_GetMemoryDevice(default_memory_info_);
  data_transfer_impl_ =
      std::make_unique<MusaDataTransfer>(ort_api, ep_api, device);

  // Create read-only allocator for use with initializers. same info as DEFAULT
  // memory apart from the allocator type. This is optional. It is only required
  // if the readonly allocator differs from the default device allocator. This
  // is not required for this cpu-based example EP, but show it as an example.
  readonly_memory_info_ =
      Ort::MemoryInfo{"MUSAExecutionProvider CPU readonly",
                      OrtMemoryInfoDeviceType_CPU,
                      /*vendor*/ 0,
                      /* device_id */ 0,
                      OrtDeviceMemoryType_DEFAULT,
                      /*alignment*/ 0,
                      OrtAllocatorType::OrtReadOnlyAllocator};
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
    // Optional state that is provided to kernels on creation (can be null).
    // We pass the OrtDataTransferImpl created by this factory to allow kernels
    // to copy data between devices.
    void* op_kernel_state =
        static_cast<OrtDataTransferImpl*>(data_transfer_impl_.get());
    const char* ep_name = ep.GetName(static_cast<const OrtEp*>(&ep));

    // This statement creates the kernel registry and caches it in the
    // OrtEpFactory instance. We assume that all EPs created by this factory can
    // use the same kernel registry. This may not be the case in a more complex
    // OrtEpFactory that can create EP instances that are each configured for
    // different hardware devices. In such a scenario, a different kernel
    // registry may be created for each EP configuration.
    RETURN_IF_ERROR(
        CreateKernelRegistry(ep_name, op_kernel_state, &kernel_registry_));
  }

  *out_kernel_registry = kernel_registry_;
  return nullptr;
}

/*static*/
const char* ORT_API_CALL
MusaEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const MusaEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL
MusaEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const MusaEpFactory*>(this_ptr);
  return factory->vendor_.c_str();
}

/*static*/
uint32_t ORT_API_CALL
MusaEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const MusaEpFactory*>(this_ptr);
  return factory->vendor_id_;
}

/*static*/
const char* ORT_API_CALL
MusaEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const MusaEpFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL MusaEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr, const OrtHardwareDevice* const* hw_devices,
    size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<MusaEpFactory*>(this_ptr);

  num_ep_devices = 0;

  int musa_device_count = 0;
  if (musaGetDeviceCount(&musa_device_count) != musaSuccess ||
      musa_device_count <= 0) {
    return nullptr;
  }

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *hw_devices[i];
    auto hw_type = factory->ort_api_.HardwareDevice_Type(&device);
    if (hw_type == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      // these can be returned as nullptr if you have nothing to add.
      OrtKeyValuePairs* ep_metadata = nullptr;
      OrtKeyValuePairs* ep_options = nullptr;
      factory->ort_api_.CreateKeyValuePairs(&ep_metadata);
      factory->ort_api_.CreateKeyValuePairs(&ep_options);

      // random example using made up values
      factory->ort_api_.AddKeyValuePair(
          ep_metadata, "supported_devices",
          std::to_string(musa_device_count).c_str());
      factory->ort_api_.AddKeyValuePair(ep_options, "device_id", "0");

      // OrtEpDevice copies ep_metadata and ep_options.
      OrtEpDevice* ep_device = nullptr;
      auto* status = factory->ort_api_.GetEpApi()->CreateEpDevice(
          factory, &device, ep_metadata, ep_options, &ep_device);

      factory->ort_api_.ReleaseKeyValuePairs(ep_metadata);
      factory->ort_api_.ReleaseKeyValuePairs(ep_options);

      if (status != nullptr) {
        return status;
      }

      // register the allocator info required by the EP.
      // registering OrtMemoryInfo for host accessible memory would be done in
      // an additional call. OrtReadOnlyAllocator + OrtDeviceMemoryType_DEFAULT
      // allocator for use with initializers is optional.
      RETURN_IF_ERROR(factory->ep_api_.EpDevice_AddAllocatorInfo(
          ep_device, factory->default_memory_info_));
      RETURN_IF_ERROR(factory->ep_api_.EpDevice_AddAllocatorInfo(
          ep_device, factory->readonly_memory_info_));

      ep_devices[num_ep_devices++] = ep_device;
    }
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL MusaEpFactory::CreateEpImpl(
    OrtEpFactory* this_ptr, const OrtHardwareDevice* const* /*devices*/,
    const OrtKeyValuePairs* const* /*ep_metadata*/, size_t num_devices,
    const OrtSessionOptions* session_options, const OrtLogger* logger,
    OrtEp** ep) noexcept {
  auto* factory = static_cast<MusaEpFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    return factory->ort_api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "MusaEpFactory only supports selection for one device.");
  }

  std::string enable_prepack_weight_sharing;
  std::string enable_cpu_preferred_metadata;
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(
      *session_options, "ep.musa.enable_prepack_weight_sharing", "0",
      enable_prepack_weight_sharing));
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(
      *session_options, "ep.musa.enable_cpu_preferred_metadata", "0",
      enable_cpu_preferred_metadata));

  MusaEp::Config config = {};
  config.enable_prepack_weight_sharing = enable_prepack_weight_sharing == "1";
  config.enable_cpu_preferred_metadata = enable_cpu_preferred_metadata == "1";

  auto actual_ep = std::make_unique<MusaEp>(*factory, config, *logger);
  *ep = actual_ep.release();

  return nullptr;
}

/*static*/
void ORT_API_CALL MusaEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/,
                                               OrtEp* ep) noexcept {
  delete static_cast<MusaEp*>(ep);
}

/*static*/
OrtStatus* ORT_API_CALL MusaEpFactory::CreateAllocatorImpl(
    OrtEpFactory* this_ptr, const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* /*allocator_options*/,
    OrtAllocator** allocator) noexcept {
  auto& factory = *static_cast<MusaEpFactory*>(this_ptr);
  *allocator = nullptr;

  bool is_default_allocator = memory_info == factory.default_memory_info_;
  bool is_readonly_allocator = memory_info == factory.readonly_memory_info_;

  if (!is_default_allocator && !is_readonly_allocator) {
    return factory.ort_api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "INTERNAL ERROR! Unknown memory info provided to CreateAllocator. "
        "Value did not come directly from an OrtEpDevice returned by this "
        "factory.");
  }

  // Note: the same allocator handles both default and readonly allocations. A
  // readonly only allocator would typically be different.
  auto custom_allocator = std::make_unique<CustomAllocator>(memory_info);
  *allocator = custom_allocator.release();
  return nullptr;
}

/*static*/
void ORT_API_CALL MusaEpFactory::ReleaseAllocatorImpl(
    OrtEpFactory* /*this_ptr*/, OrtAllocator* allocator) noexcept {
  delete static_cast<CustomAllocator*>(allocator);
}

/*static*/
OrtStatus* ORT_API_CALL MusaEpFactory::CreateDataTransferImpl(
    OrtEpFactory* this_ptr, OrtDataTransferImpl** data_transfer) noexcept {
  auto& factory = *static_cast<MusaEpFactory*>(this_ptr);
  *data_transfer = factory.data_transfer_impl_.get();
  return nullptr;
}

/*static*/
bool ORT_API_CALL
MusaEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return true;
}

/*static*/
OrtStatus* ORT_API_CALL MusaEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* this_ptr, const OrtMemoryDevice* memory_device,
    const OrtKeyValuePairs* /*stream_opts*/,
    OrtSyncStreamImpl** stream) noexcept {
  auto& factory = *static_cast<MusaEpFactory*>(this_ptr);
  *stream = nullptr;

  if (factory.ep_api_.MemoryDevice_GetMemoryType(memory_device) !=
      OrtDeviceMemoryType_DEFAULT) {
    return nullptr;
  }

  try {
    auto sync_stream = std::make_unique<MusaSyncStream>(factory.ort_api_);
    *stream = sync_stream.release();
    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return factory.ort_api_.CreateStatus(ORT_EP_FAIL, ex.what());
  }
}
