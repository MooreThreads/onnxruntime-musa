#include "ep_data_transfer.h"

#include <musa_runtime.h>

#include <span>

namespace {
bool IsGpu(const OrtEpApi& ep_api, const OrtMemoryDevice* device) {
  return ep_api.MemoryDevice_GetDeviceType(device) ==
         OrtMemoryInfoDeviceType_GPU;
}

OrtStatus* CopyImpl(const OrtApi& api, const OrtEpApi& ep_api,
                    const OrtMemoryDevice* src_device,
                    const OrtMemoryDevice* dst_device, const void* src_data,
                    void* dst_data, size_t bytes) {
  if (bytes == 0 || src_data == dst_data) {
    return nullptr;
  }

  const bool src_gpu = IsGpu(ep_api, src_device);
  const bool dst_gpu = IsGpu(ep_api, dst_device);

  musaMemcpyKind kind = musaMemcpyHostToHost;
  if (src_gpu && dst_gpu) {
    kind = musaMemcpyDeviceToDevice;
  } else if (src_gpu) {
    kind = musaMemcpyDeviceToHost;
  } else if (dst_gpu) {
    kind = musaMemcpyHostToDevice;
  }

  musaError_t status = musaMemcpy(dst_data, src_data, bytes, kind);
  if (status != musaSuccess) {
    return api.CreateStatus(ORT_EP_FAIL, musaGetErrorString(status));
  }
  return nullptr;
}
}  // namespace

bool ORT_API_CALL MusaDataTransfer::CanCopyImpl(
    const OrtDataTransferImpl* this_ptr,
    const OrtMemoryDevice* src_memory_device,
    const OrtMemoryDevice* dst_memory_device) noexcept {
  const auto& impl = *static_cast<const MusaDataTransfer*>(this_ptr);
  bool src_is_our_device = impl.ep_api_.MemoryDevice_AreEqual(
      src_memory_device, impl.device_mem_info);
  bool dst_is_our_device = impl.ep_api_.MemoryDevice_AreEqual(
      dst_memory_device, impl.device_mem_info);

  if (src_is_our_device || dst_is_our_device) {
    return true;
  }

  return false;
}

OrtStatus* ORT_API_CALL MusaDataTransfer::CopyTensorsImpl(
    OrtDataTransferImpl* this_ptr, const OrtValue** src_tensors_ptr,
    OrtValue** dst_tensors_ptr, OrtSyncStream** /*streams_ptr*/,
    size_t num_tensors) noexcept {
  auto& impl = *static_cast<MusaDataTransfer*>(this_ptr);

  auto src_tensors =
      std::span<const OrtValue* const>(src_tensors_ptr, num_tensors);
  auto dst_tensors = std::span<OrtValue*>(dst_tensors_ptr, num_tensors);

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
    RETURN_IF_ERROR(CopyImpl(impl.ort_api_, impl.ep_api_, src_device,
                             dst_device, src_data, dst_data, bytes));
  }

  return nullptr;
}

void ORT_API_CALL
MusaDataTransfer::ReleaseImpl(OrtDataTransferImpl* /*this_ptr*/) noexcept {}
