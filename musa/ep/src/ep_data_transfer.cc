#include "ep_data_transfer.h"

#include <musa_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <execinfo.h>
#include <unistd.h>
#endif

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
  return !(text == "0" || text == "false" || text == "FALSE" || text == "off" ||
           text == "OFF");
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

// The plugin data-transfer ABI deliberately exposes OrtValue storage and
// memory devices, but not the producing node/value-info names. Keep this
// diagnostic opt-in and print stable pointer/size keys that can be joined
// with an ORT/MTTX trace. The backtrace identifies the exact transfer path.
void TraceD2hDeviceSynchronize(const OrtValue* src_tensor,
                               const OrtValue* dst_tensor, const void* src_data,
                               void* dst_data, size_t bytes,
                               size_t tensor_index, size_t tensor_count) {
  if (!EnvFlagEnabled("ORT_MUSA_TRACE_D2H_SYNC", false)) return;
  static std::atomic<uint64_t> sequence{0};
  const uint64_t id = ++sequence;
  std::fprintf(stderr,
               "MUSA_D2H_DEVICE_SYNC id=%llu src_value=%p dst_value=%p "
               "src_data=%p dst_data=%p bytes=%zu stream=null "
               "tensor_index=%zu tensor_count=%zu "
               "node=<data-transfer-abi-no-node-name>\n",
               static_cast<unsigned long long>(id),
               static_cast<const void*>(src_tensor),
               static_cast<const void*>(dst_tensor), src_data, dst_data, bytes,
               tensor_index, tensor_count);
#if defined(__linux__) || defined(__APPLE__)
  void* frames[32];
  const int count = ::backtrace(frames, 32);
  std::fprintf(stderr, "MUSA_D2H_DEVICE_SYNC_BACKTRACE id=%llu frames=%d\n",
               static_cast<unsigned long long>(id), count);
  ::backtrace_symbols_fd(frames, count, STDERR_FILENO);
#else
  std::fprintf(stderr, "MUSA_D2H_DEVICE_SYNC_BACKTRACE id=%llu unavailable\n",
               static_cast<unsigned long long>(id));
#endif
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
    return MusaStatus(impl.ort_api_, musaMemcpy(dst_data, src_data, bytes,
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
      return CopyPageableHostToDevice(impl, src_data, dst_data, bytes, stream,
                                      allow_pageable_bounce);
    }

    return CopyMemcpy(impl.ort_api_, src_data, dst_data, bytes,
                      musaMemcpyHostToDevice, stream);
  }

  if (src_is_gpu_default) {
    // ORT normally supplies the producer stream for device-to-host metadata
    // copies. Synchronizing that stream preserves request-local ordering
    // without stalling unrelated workers. Some CPU fallback edges do not
    // carry a stream, though; a synchronous D2H copy alone does not order work
    // submitted to non-blocking MUSA streams, so retain the device-wide
    // correctness barrier only for that fallback case.
    if (stream == nullptr) {
      RETURN_IF_ERROR(MusaStatus(impl.ort_api_, musaDeviceSynchronize()));
    }
    RETURN_IF_ERROR(CopyMemcpy(impl.ort_api_, src_data, dst_data, bytes,
                               musaMemcpyDeviceToHost, stream));
    if (stream != nullptr) {
      RETURN_IF_ERROR(MusaStatus(impl.ort_api_, musaStreamSynchronize(stream)));
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

struct TensorCopy {
  const OrtValue* src_tensor = nullptr;
  OrtValue* dst_tensor = nullptr;
  const OrtMemoryDevice* src_device = nullptr;
  const OrtMemoryDevice* dst_device = nullptr;
  const void* src_data = nullptr;
  void* dst_data = nullptr;
  size_t bytes = 0;
  musaStream_t stream = nullptr;
};

bool CanBatchDeviceToHost(const MusaDataTransfer& impl,
                          std::span<const TensorCopy> copies,
                          musaStream_t& batch_stream) {
  size_t active_copies = 0;
  for (const TensorCopy& copy : copies) {
    if (copy.bytes == 0 || copy.src_data == copy.dst_data) {
      continue;
    }
    if (!IsGpuDefault(impl.ep_api_, copy.src_device, impl.vendor_id_) ||
        IsGpuDefault(impl.ep_api_, copy.dst_device, impl.vendor_id_) ||
        copy.stream == nullptr) {
      return false;
    }
    if (active_copies == 0) {
      batch_stream = copy.stream;
    } else if (copy.stream != batch_stream) {
      return false;
    }
    ++active_copies;
  }
  return active_copies != 0;
}

OrtStatus* CopyDeviceToHostBatch(MusaDataTransfer& impl,
                                 std::span<const TensorCopy> copies,
                                 musaStream_t stream) {
  size_t staging_bytes = 0;
  for (const TensorCopy& copy : copies) {
    if (copy.bytes != 0 && copy.src_data != copy.dst_data &&
        !IsHostAccessible(impl.ep_api_, copy.dst_device, impl.vendor_id_)) {
      staging_bytes += copy.bytes;
    }
  }

  void* staging = staging_bytes != 0 && impl.pinned_host_pool_ != nullptr
                      ? impl.pinned_host_pool_->Allocate(staging_bytes)
                      : nullptr;
  auto* staging_data = static_cast<uint8_t*>(staging);
  std::vector<size_t> staging_offsets(copies.size(), 0);
  size_t next_offset = 0;
  size_t enqueued = 0;
  for (size_t i = 0; i < copies.size(); ++i) {
    const TensorCopy& copy = copies[i];
    if (copy.bytes == 0 || copy.src_data == copy.dst_data) {
      continue;
    }

    const bool needs_staging =
        staging != nullptr &&
        !IsHostAccessible(impl.ep_api_, copy.dst_device, impl.vendor_id_);
    void* copy_dst = copy.dst_data;
    if (needs_staging) {
      staging_offsets[i] = next_offset;
      copy_dst = staging_data + next_offset;
      next_offset += copy.bytes;
    }

    musaError_t status = musaMemcpyAsync(copy_dst, copy.src_data, copy.bytes,
                                         musaMemcpyDeviceToHost, stream);
    if (status != musaSuccess) {
      if (enqueued != 0) {
        (void)musaStreamSynchronize(stream);
      }
      if (staging != nullptr) {
        impl.pinned_host_pool_->FreeCompleted(staging);
      }
      return MusaStatus(impl.ort_api_, status);
    }
    ++enqueued;
  }

  musaError_t status = musaStreamSynchronize(stream);
  if (status != musaSuccess) {
    if (staging != nullptr) {
      impl.pinned_host_pool_->FreeAsync(staging, stream);
    }
    return MusaStatus(impl.ort_api_, status);
  }

  if (staging != nullptr) {
    for (size_t i = 0; i < copies.size(); ++i) {
      const TensorCopy& copy = copies[i];
      if (copy.bytes != 0 && copy.src_data != copy.dst_data &&
          !IsHostAccessible(impl.ep_api_, copy.dst_device, impl.vendor_id_)) {
        std::memcpy(copy.dst_data, staging_data + staging_offsets[i],
                    copy.bytes);
      }
    }
    impl.pinned_host_pool_->FreeCompleted(staging);
  }
  return nullptr;
}

}  // namespace

bool ORT_API_CALL MusaDataTransfer::CanCopyImpl(
    const OrtDataTransferImpl* this_ptr,
    const OrtMemoryDevice* src_memory_device,
    const OrtMemoryDevice* dst_memory_device) noexcept {
  const auto& impl = *static_cast<const MusaDataTransfer*>(this_ptr);
  const bool src_is_our_device =
      impl.ep_api_.MemoryDevice_AreEqual(src_memory_device,
                                         impl.device_mem_info) ||
      (impl.host_accessible_mem_info != nullptr &&
       impl.ep_api_.MemoryDevice_AreEqual(src_memory_device,
                                          impl.host_accessible_mem_info));
  const bool dst_is_our_device =
      impl.ep_api_.MemoryDevice_AreEqual(dst_memory_device,
                                         impl.device_mem_info) ||
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
      num_tensors >=
      EnvSizeOrDefault("ORT_MUSA_PAGEABLE_H2D_BOUNCE_MIN_TENSORS", 1024);

  std::vector<TensorCopy> copies;
  copies.reserve(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    TensorCopy copy;
    copy.src_tensor = src_tensors[i];
    copy.dst_tensor = dst_tensors[i];
    copy.src_device = impl.ep_api_.Value_GetMemoryDevice(src_tensors[i]);
    copy.dst_device = impl.ep_api_.Value_GetMemoryDevice(dst_tensors[i]);

    RETURN_IF_ERROR(
        impl.ort_api_.GetTensorData(src_tensors[i], &copy.src_data));
    RETURN_IF_ERROR(
        impl.ort_api_.GetTensorMutableData(dst_tensors[i], &copy.dst_data));
    RETURN_IF_ERROR(
        impl.ort_api_.GetTensorSizeInBytes(src_tensors[i], &copy.bytes));

    if (streams_ptr != nullptr && streams_ptr[i] != nullptr) {
      copy.stream = static_cast<musaStream_t>(
          impl.ort_api_.SyncStream_GetHandle(streams_ptr[i]));
    }

    if (copy.bytes != 0 && copy.src_data != copy.dst_data &&
        copy.stream == nullptr &&
        IsGpuDefault(impl.ep_api_, copy.src_device, impl.vendor_id_) &&
        !IsGpuDefault(impl.ep_api_, copy.dst_device, impl.vendor_id_)) {
      TraceD2hDeviceSynchronize(copy.src_tensor, copy.dst_tensor, copy.src_data,
                                copy.dst_data, copy.bytes, i, num_tensors);
    }
    copies.push_back(copy);
  }

  musaStream_t batch_stream = nullptr;
  if (CanBatchDeviceToHost(impl, copies, batch_stream)) {
    return CopyDeviceToHostBatch(impl, copies, batch_stream);
  }

  for (const TensorCopy& copy : copies) {
    RETURN_IF_ERROR(CopyImpl(impl, copy.src_device, copy.dst_device,
                             copy.src_data, copy.dst_data, copy.bytes,
                             copy.stream, allow_pageable_bounce));
  }

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

void ORT_API_CALL
MusaDataTransfer::ReleaseImpl(OrtDataTransferImpl* /*this_ptr*/) noexcept {}
