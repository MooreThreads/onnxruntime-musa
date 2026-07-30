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

#include <mutex>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "ep_data_transfer.h"
#include "pinned_host_pool.h"

class MusaEp;

/// <summary>
/// EP factory that creates an OrtEp instance that uses kernel registration.
/// </summary>
class MusaEpFactory : public OrtEpFactory {
 public:
  MusaEpFactory(const OrtApi& ort_api, const OrtEpApi& ep_api,
                const OrtLogger& default_logger);
  ~MusaEpFactory();

  const OrtApi& GetOrtApi() const { return ort_api_; }
  const OrtEpApi& GetEpApi() const { return ep_api_; }
  const std::string& GetEpName() const { return ep_name_; }
  uint32_t VendorId() const { return vendor_id_; }

  // Called by child OrtEp instances to retrieve the cached kernel registry for
  // that EP.
  OrtStatus* GetKernelRegistryForEp(
      MusaEp& ep, /*out*/ const OrtKernelRegistry** kernel_registry);

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
      size_t* p_num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(
      OrtEpFactory* this_ptr, const OrtHardwareDevice* const* /*devices*/,
      const OrtKeyValuePairs* const* /*ep_metadata*/, size_t num_devices,
      const OrtSessionOptions* session_options, const OrtLogger* logger,
      OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/,
                                         OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL
  CreateAllocatorImpl(OrtEpFactory* this_ptr, const OrtMemoryInfo* memory_info,
                      const OrtKeyValuePairs* /*allocator_options*/,
                      OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(
      OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(
      OrtEpFactory* this_ptr, OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL
  IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEpFactory* this_ptr, const OrtMemoryDevice* memory_device,
      const OrtKeyValuePairs* stream_options,
      OrtSyncStreamImpl** stream) noexcept;

  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  const OrtLogger& default_logger_;
  const std::string ep_name_{"MUSAExecutionProvider"};
  const std::string vendor_{"MThreads"};  // EP vendor name
  const uint32_t vendor_id_{0x4D54};      // EP vendor ID
#ifndef ONNXRUNTIME_MUSA_VERSION
#define ONNXRUNTIME_MUSA_VERSION "0.0.0"
#endif
  const std::string ep_version_{
      ONNXRUNTIME_MUSA_VERSION};  // EP version (injected from VERSION_NUMBER)

  Ort::MemoryInfo default_memory_info_;
  Ort::MemoryInfo host_accessible_memory_info_;
  Ort::MemoryInfo readonly_memory_info_;
  std::unique_ptr<MusaDataTransfer>
      data_transfer_impl_;  // data transfer implementation for this factory
  std::shared_ptr<PinnedHostPool> pinned_host_pool_;

  // Cached kernel registry used by all OrtEp instances created by this factory.
  // Refer to OrtEp::GetKernelRegistry.
  //
  // Note: If this factory instead created EP instances that each supported
  // different hardware configurations, then the factory could cache a different
  // kernel registry per EP configuration.
  OrtKernelRegistry* kernel_registry_ = nullptr;
};
