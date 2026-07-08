// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "ep_stream.h"

#include <memory>

#include "ep_allocator.h"

namespace {

OrtStatus* MusaStatus(const OrtApi& ort_api, musaError_t status) {
  if (status == musaSuccess) {
    return nullptr;
  }
  return ort_api.CreateStatus(ORT_EP_FAIL, musaGetErrorString(status));
}

class MusaNotification : public OrtSyncNotificationImpl {
 public:
  MusaNotification(const OrtApi& ort_api, musaStream_t producer_stream)
      : ort_api_(ort_api), producer_stream_(producer_stream) {
    ort_version_supported = ORT_API_VERSION;
    Activate = ActivateImpl;
    WaitOnDevice = WaitOnDeviceImpl;
    WaitOnHost = WaitOnHostImpl;
    Release = ReleaseImpl;

    musaError_t status =
        musaEventCreateWithFlags(&event_, musaEventDisableTiming);
    if (status != musaSuccess) {
      event_ = nullptr;
      create_error_ = status;
    }
  }

  ~MusaNotification() {
    if (event_ != nullptr) {
      (void)musaEventDestroy(event_);
    }
  }

 private:
  static OrtStatus* ORT_API_CALL
  ActivateImpl(OrtSyncNotificationImpl* this_ptr) noexcept {
    auto& impl = *static_cast<MusaNotification*>(this_ptr);
    if (impl.create_error_ != musaSuccess) {
      return MusaStatus(impl.ort_api_, impl.create_error_);
    }
    return MusaStatus(impl.ort_api_,
                      musaEventRecord(impl.event_, impl.producer_stream_));
  }

  static OrtStatus* ORT_API_CALL WaitOnDeviceImpl(
      OrtSyncNotificationImpl* this_ptr, OrtSyncStream* stream) noexcept {
    auto& impl = *static_cast<MusaNotification*>(this_ptr);
    if (impl.create_error_ != musaSuccess) {
      return MusaStatus(impl.ort_api_, impl.create_error_);
    }
    if (stream == nullptr) {
      return nullptr;
    }

    musaStream_t consumer_stream =
        static_cast<musaStream_t>(impl.ort_api_.SyncStream_GetHandle(stream));
    return MusaStatus(impl.ort_api_,
                      musaStreamWaitEvent(consumer_stream, impl.event_, 0));
  }

  static OrtStatus* ORT_API_CALL
  WaitOnHostImpl(OrtSyncNotificationImpl* this_ptr) noexcept {
    auto& impl = *static_cast<MusaNotification*>(this_ptr);
    if (impl.create_error_ != musaSuccess) {
      return MusaStatus(impl.ort_api_, impl.create_error_);
    }
    return MusaStatus(impl.ort_api_, musaEventSynchronize(impl.event_));
  }

  static void ORT_API_CALL
  ReleaseImpl(OrtSyncNotificationImpl* this_ptr) noexcept {
    delete static_cast<MusaNotification*>(this_ptr);
  }

  const OrtApi& ort_api_;
  musaStream_t producer_stream_ = nullptr;
  musaEvent_t event_ = nullptr;
  musaError_t create_error_ = musaSuccess;
};

}  // namespace

MusaSyncStream::MusaSyncStream(const OrtApi& ort_api) : ort_api_(ort_api) {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  GetHandle = GetHandleImpl;
  CreateNotification = CreateNotificationImpl;
  Flush = FlushImpl;
  OnSessionRunEnd = OnSessionRunEndImpl;

  musaError_t status =
      musaStreamCreateWithFlags(&stream_, musaStreamNonBlocking);
  if (status != musaSuccess) {
    stream_ = nullptr;
    throw Ort::Exception(musaGetErrorString(status), ORT_EP_FAIL);
  }
}

MusaSyncStream::MusaSyncStream(const OrtApi& ort_api,
                               musaStream_t external_stream)
    : ort_api_(ort_api), stream_(external_stream), owns_stream_(false) {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  GetHandle = GetHandleImpl;
  CreateNotification = CreateNotificationImpl;
  Flush = FlushImpl;
  OnSessionRunEnd = OnSessionRunEndImpl;
  if (stream_ == nullptr) {
    throw Ort::Exception("MUSA external compute stream must not be null.",
                         ORT_INVALID_ARGUMENT);
  }
}

MusaSyncStream::~MusaSyncStream() {
  CustomAllocator::ResetBlocksUsingStream(this);
  if (owns_stream_ && stream_ != nullptr) {
    (void)musaStreamDestroy(stream_);
  }
}

void ORT_API_CALL
MusaSyncStream::ReleaseImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  delete static_cast<MusaSyncStream*>(this_ptr);
}

void* ORT_API_CALL
MusaSyncStream::GetHandleImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  return static_cast<MusaSyncStream*>(this_ptr)->stream();
}

OrtStatus* ORT_API_CALL MusaSyncStream::CreateNotificationImpl(
    OrtSyncStreamImpl* this_ptr,
    OrtSyncNotificationImpl** notification) noexcept {
  auto& impl = *static_cast<MusaSyncStream*>(this_ptr);
  *notification =
      std::make_unique<MusaNotification>(impl.ort_api_, impl.stream())
          .release();
  return nullptr;
}

OrtStatus* ORT_API_CALL
MusaSyncStream::FlushImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  auto& impl = *static_cast<MusaSyncStream*>(this_ptr);
  return MusaStatus(impl.ort_api_, musaStreamSynchronize(impl.stream()));
}

OrtStatus* ORT_API_CALL
MusaSyncStream::OnSessionRunEndImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  (void)this_ptr;
  // ORT calls Flush separately when a synchronized Run requires host-visible
  // completion. Keep same-stream allocator caches intact across runs.
  return nullptr;
}
