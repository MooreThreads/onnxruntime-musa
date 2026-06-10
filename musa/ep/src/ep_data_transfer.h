// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "plugin_ep_utils.h"
#include "pinned_host_pool.h"

#include <memory>
#include <utility>

struct MusaDataTransfer : OrtDataTransferImpl {
  MusaDataTransfer(const OrtApi& ort_api, const OrtEpApi& ep_api,
                   const OrtMemoryDevice* device_mem_info_,
                   const OrtMemoryDevice* host_accessible_mem_info_,
                   uint32_t vendor_id,
                   std::shared_ptr<PinnedHostPool> pinned_host_pool)
      : ort_api_(ort_api),
        ep_api_(ep_api),
        device_mem_info{device_mem_info_},
        host_accessible_mem_info{host_accessible_mem_info_},
        vendor_id_{vendor_id},
        pinned_host_pool_{std::move(pinned_host_pool)} {
    CanCopy = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
    Release = ReleaseImpl;
  }

  static bool ORT_API_CALL CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                       const OrtMemoryDevice* src_memory_device,
                                       const OrtMemoryDevice* dst_memory_device) noexcept;

  // function to copy one or more tensors.
  // implementation can optionally use async copy if a stream is available for the input.
  static OrtStatus* ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                 const OrtValue** src_tensors_ptr,
                                                 OrtValue** dst_tensors_ptr,
                                                 OrtSyncStream** streams_ptr,
                                                 size_t num_tensors) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept;

  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  const OrtMemoryDevice* device_mem_info;  // device our EP runs on
  const OrtMemoryDevice* host_accessible_mem_info;
  uint32_t vendor_id_;
  std::shared_ptr<PinnedHostPool> pinned_host_pool_;
};
