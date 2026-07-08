// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

class MusaSyncStream : public OrtSyncStreamImpl {
 public:
  explicit MusaSyncStream(const OrtApi& ort_api);
  MusaSyncStream(const OrtApi& ort_api, musaStream_t external_stream);
  ~MusaSyncStream();

  MusaSyncStream(const MusaSyncStream&) = delete;
  MusaSyncStream& operator=(const MusaSyncStream&) = delete;

  musaStream_t stream() const { return stream_; }

 private:
  static void ORT_API_CALL ReleaseImpl(OrtSyncStreamImpl* this_ptr) noexcept;
  static void* ORT_API_CALL GetHandleImpl(OrtSyncStreamImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL
  CreateNotificationImpl(OrtSyncStreamImpl* this_ptr,
                         OrtSyncNotificationImpl** notification) noexcept;
  static OrtStatus* ORT_API_CALL
  FlushImpl(OrtSyncStreamImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL
  OnSessionRunEndImpl(OrtSyncStreamImpl* this_ptr) noexcept;

  const OrtApi& ort_api_;
  musaStream_t stream_ = nullptr;
  bool owns_stream_ = true;
};
