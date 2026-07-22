#include "ep_data_transfer.h"

#include <musa_runtime.h>

#include <cstdlib>
#include <cstring>
#include <span>
#include <string_view>

namespace {
OrtStatus* MusaStatus(const OrtApi& api, musaError_t status) {
  if (status == musaSuccess) {
    return nullptr;
  }

  return api.CreateStatus(ORT_EP_FAIL, musaGetErrorString(status));
}

bool IsGpuDefault(const OrtEpApi& ep_api, const OrtMemoryDevice* device,
                  uint32_t vendor_id) {
  return ep_api.MemoryDevice_GetDeviceType(device) ==
             OrtMemoryInfoDeviceType_GPU &&
         ep_api.MemoryDevice_GetMemoryType(device) ==
             OrtDeviceMemoryType_DEFAULT &&
         ep_api.MemoryDevice_GetVendorId(device) == vendor_id;
}

bool IsHostAccessible(const OrtEpApi& ep_api, const OrtMemoryDevice* device,
                      uint32_t vendor_id) {
  return ep_api.MemoryDevice_GetMemoryType(device) ==
             OrtDeviceMemoryType_HOST_ACCESSIBLE &&
         ep_api.MemoryDevice_GetVendorId(device) == vendor_id;
}

OrtStatus* CopyMemcpy(const OrtApi& api, const void* src_data, void* dst_data,
                      size_t bytes, musaMemcpyKind kind, musaStream_t stream) {
  if (stream != nullptr) {
    return MusaStatus(api,
                      musaMemcpyAsync(dst_data, src_data, bytes, kind, stream));
  }

  return MusaStatus(api, musaMemcpy(dst_data, src_data, bytes, kind));
}

bool EnvFlagEnabled(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return default_value;
  }

  std::string_view text{value};
  return !(text == "0" || text == "false" || text == "FALSE" ||
           text == "off" || text == "OFF");
}

size_t EnvSizeOrDefault(const char* name, size_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return default_value;
  }

  char* end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value) {
    return default_value;
  }
  return static_cast<size_t>(parsed);
}

OrtStatus* CopyPageableHostToDevice(MusaDataTransfer& impl,
                                    const void* src_data, void* dst_data,
                                    size_t bytes, musaStream_t stream,
                                    bool allow_pageable_bounce) {
  constexpr size_t kDefaultBounceThresholdBytes = 1024;
  const size_t bounce_threshold =
      EnvSizeOrDefault("ORT_MUSA_PAGEABLE_H2D_BOUNCE_THRESHOLD_BYTES",
                       kDefaultBounceThresholdBytes);
  if (bytes < bounce_threshold) {
    return MusaStatus(impl.ort_api_,
                      musaMemcpy(dst_data, src_data, bytes,
                                 musaMemcpyHostToDevice));
  }

  if (!allow_pageable_bounce || stream == nullptr ||
      impl.pinned_host_pool_ == nullptr ||
      !EnvFlagEnabled("ORT_MUSA_ENABLE_PAGEABLE_H2D_BOUNCE", true)) {
    return CopyMemcpy(impl.ort_api_, src_data, dst_data, bytes,
                      musaMemcpyHostToDevice, stream);
  }

  void* staging = impl.pinned_host_pool_->Allocate(bytes);
  if (staging == nullptr) {
    return CopyMemcpy(impl.ort_api_, src_data, dst_data, bytes,
                      musaMemcpyHostToDevice, stream);
  }

  std::memcpy(staging, src_data, bytes);
  musaError_t status =
      musaMemcpyAsync(dst_data, staging, bytes, musaMemcpyHostToDevice, stream);
  if (status != musaSuccess) {
    impl.pinned_host_pool_->FreeCompleted(staging);
    return MusaStatus(impl.ort_api_, status);
  }

  impl.pinned_host_pool_->FreeAsync(staging, stream);
  return nullptr;
}

OrtStatus* CopyImpl(MusaDataTransfer& impl, const OrtMemoryDevice* src_device,
                    const OrtMemoryDevice* dst_device, const void* src_data,
                    void* dst_data, size_t bytes, musaStream_t stream,
                    bool allow_pageable_bounce) {
  if (bytes == 0 || src_data == dst_data) {
    return nullptr;
  }

  const bool src_is_gpu_default =
      IsGpuDefault(impl.ep_api_, src_device, impl.vendor_id_);
  const bool dst_is_gpu_default =
      IsGpuDefault(impl.ep_api_, dst_device, impl.vendor_id_);

  if (dst_is_gpu_default) {
    if (src_is_gpu_default) {
      return CopyMemcpy(impl.ort_api_, src_data, dst_data, bytes,
                        musaMemcpyDeviceToDevice, stream);
    }

    if (!IsHostAccessible(impl.ep_api_, src_device, impl.vendor_id_)) {
      return CopyPageableHostToDevice(impl, src_data, dst_data, bytes,
                                      stream, allow_pageable_bounce);
    }

    return CopyMemcpy(impl.ort_api_, src_data, dst_data, bytes,
                      musaMemcpyHostToDevice, stream);
  }

  if (src_is_gpu_default) {
    RETURN_IF_ERROR(MusaStatus(impl.ort_api_, musaDeviceSynchronize()));
    RETURN_IF_ERROR(CopyMemcpy(impl.ort_api_, src_data, dst_data, bytes,
                               musaMemcpyDeviceToHost, stream));
    if (stream != nullptr) {
      RETURN_IF_ERROR(
          MusaStatus(impl.ort_api_, musaStreamSynchronize(stream)));
    }
    return nullptr;
  }

  if (stream != nullptr &&
      IsHostAccessible(impl.ep_api_, src_device, impl.vendor_id_)) {
    RETURN_IF_ERROR(MusaStatus(impl.ort_api_, musaStreamSynchronize(stream)));
  }

  std::memcpy(dst_data, src_data, bytes);
  return nullptr;
}

}  // namespace

bool ORT_API_CALL MusaDataTransfer::CanCopyImpl(
    const OrtDataTransferImpl* this_ptr,
    const OrtMemoryDevice* src_memory_device,
    const OrtMemoryDevice* dst_memory_device) noexcept {
  const auto& impl = *static_cast<const MusaDataTransfer*>(this_ptr);
  const bool src_is_our_device =
      impl.ep_api_.MemoryDevice_AreEqual(src_memory_device, impl.device_mem_info) ||
      (impl.host_accessible_mem_info != nullptr &&
       impl.ep_api_.MemoryDevice_AreEqual(src_memory_device,
                                          impl.host_accessible_mem_info));
  const bool dst_is_our_device =
      impl.ep_api_.MemoryDevice_AreEqual(dst_memory_device, impl.device_mem_info) ||
      (impl.host_accessible_mem_info != nullptr &&
       impl.ep_api_.MemoryDevice_AreEqual(dst_memory_device,
                                          impl.host_accessible_mem_info));

  return src_is_our_device || dst_is_our_device;
}

OrtStatus* ORT_API_CALL MusaDataTransfer::CopyTensorsImpl(
    OrtDataTransferImpl* this_ptr, const OrtValue** src_tensors_ptr,
    OrtValue** dst_tensors_ptr, OrtSyncStream** streams_ptr,
    size_t num_tensors) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto& impl = *static_cast<MusaDataTransfer*>(this_ptr);

  auto src_tensors =
      std::span<const OrtValue* const>(src_tensors_ptr, num_tensors);
  auto dst_tensors = std::span<OrtValue*>(dst_tensors_ptr, num_tensors);
  const bool allow_pageable_bounce =
      num_tensors >= EnvSizeOrDefault(
                         "ORT_MUSA_PAGEABLE_H2D_BOUNCE_MIN_TENSORS", 1024);

  for (size_t i = 0; i < num_tensors; ++i) {
    const OrtMemoryDevice* src_device =
        impl.ep_api_.Value_GetMemoryDevice(src_tensors[i]);
    const OrtMemoryDevice* dst_device =
        impl.ep_api_.Value_GetMemoryDevice(dst_tensors[i]);

    const void* src_data = nullptr;
    void* dst_data = nullptr;
    size_t bytes = 0;

    RETURN_IF_ERROR(impl.ort_api_.GetTensorData(src_tensors[i], &src_data));
    RETURN_IF_ERROR(
        impl.ort_api_.GetTensorMutableData(dst_tensors[i], &dst_data));
    RETURN_IF_ERROR(impl.ort_api_.GetTensorSizeInBytes(src_tensors[i], &bytes));

    musaStream_t stream = nullptr;
    if (streams_ptr != nullptr && streams_ptr[i] != nullptr) {
      stream = static_cast<musaStream_t>(
          impl.ort_api_.SyncStream_GetHandle(streams_ptr[i]));
    }

    RETURN_IF_ERROR(CopyImpl(impl, src_device, dst_device, src_data, dst_data,
                             bytes, stream, allow_pageable_bounce));
  }

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

void ORT_API_CALL
MusaDataTransfer::ReleaseImpl(OrtDataTransferImpl* /*this_ptr*/) noexcept {}
