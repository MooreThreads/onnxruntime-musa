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
