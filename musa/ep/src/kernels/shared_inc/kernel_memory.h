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

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "pinned_host_pool.h"
#include "runtime/ep_musa_utils.h"
#include "utils.h"

inline musaStream_t GetComputeStream(const Ort::KernelContext& ctx) {
  return static_cast<musaStream_t>(ctx.GetGPUComputeStream());
}

inline OrtStatus* WaitForDefaultStream(musaStream_t stream) {
  if (stream == nullptr) {
    return nullptr;
  }

  musaEvent_t event = nullptr;
  musaError_t status = musaEventCreateWithFlags(&event, musaEventDisableTiming);
  if (status != musaSuccess) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
  }

  status = musaEventRecord(event, nullptr);
  if (status == musaSuccess) {
    status = musaStreamWaitEvent(stream, event, 0);
  }

  (void)musaEventDestroy(event);
  if (status != musaSuccess) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
  }
  return nullptr;
}

inline OrtStatus* WaitForDefaultStreamBeforeHostCopy(musaStream_t stream) {
  return WaitForDefaultStream(stream);
}

class DeferredDeviceFreeQueue {
 public:
  DeferredDeviceFreeQueue()
      : worker_(&DeferredDeviceFreeQueue::PollLoop, this) {}

  ~DeferredDeviceFreeQueue() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      cv_.notify_all();
    }
    if (worker_.joinable()) {
      worker_.join();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    for (PendingFree& pending : pending_) {
      if (pending.event != nullptr) {
        while (musaEventQuery(pending.event) == musaErrorNotReady) {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        (void)musaEventDestroy(pending.event);
      }
      (void)musaFree(pending.ptr);
    }
    // Keep completed cached blocks until process exit. Freeing device memory
    // from static teardown can race MUSA runtime destruction.
    cached_.clear();
    cached_bytes_ = 0;
  }

  void* Allocate(size_t bytes, musaStream_t stream) {
    if (bytes == 0) {
      return nullptr;
    }

    const size_t rounded_bytes = RoundSize(bytes);
    (void)stream;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = cached_.lower_bound(rounded_bytes); it != cached_.end();
           ++it) {
        void* ptr = it->second.ptr;
        cached_bytes_ -= it->first;
        cached_.erase(it);
        return ptr;
      }
    }

    void* ptr = nullptr;
    return musaMalloc(&ptr, rounded_bytes) == musaSuccess ? ptr : nullptr;
  }

  void Free(void* ptr, musaStream_t stream, size_t bytes = 0) {
    if (ptr == nullptr) {
      return;
    }
    const size_t rounded_bytes = bytes == 0 ? 0 : RoundSize(bytes);
    if (stream == nullptr) {
      (void)musaFree(ptr);
      return;
    }

    musaEvent_t event = nullptr;
    if (musaEventCreateWithFlags(&event, musaEventDisableTiming) !=
        musaSuccess) {
      (void)musaStreamSynchronize(stream);
      (void)musaFree(ptr);
      return;
    }
    if (musaEventRecord(event, stream) != musaSuccess) {
      (void)musaEventDestroy(event);
      (void)musaStreamSynchronize(stream);
      (void)musaFree(ptr);
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back({ptr, event, rounded_bytes});
    cv_.notify_one();
  }

 private:
  struct PendingFree {
    void* ptr = nullptr;
    musaEvent_t event = nullptr;
    size_t bytes = 0;
  };

  struct CachedBlock {
    void* ptr = nullptr;
  };

  static size_t RoundSize(size_t bytes) {
    constexpr size_t kAlignment = 256;
    return (bytes + kAlignment - 1) & ~(kAlignment - 1);
  }

  static size_t CacheLimitBytes() {
    const char* env = std::getenv("ORT_MUSA_DEFERRED_FREE_CACHE_LIMIT_MB");
    if (env != nullptr && *env != '\0') {
      long mb = std::strtol(env, nullptr, 10);
      return mb <= 0 ? 0 : static_cast<size_t>(mb) * 1024 * 1024;
    }
    return 256 * 1024 * 1024;
  }

  bool CacheCompleted(PendingFree& pending) {
    if (pending.bytes == 0) {
      return false;
    }
    const size_t limit = CacheLimitBytes();
    if (limit == 0 || cached_bytes_ + pending.bytes > limit) {
      return false;
    }

    cached_.emplace(pending.bytes, CachedBlock{pending.ptr});
    cached_bytes_ += pending.bytes;
    pending.ptr = nullptr;
    return true;
  }

  void PollLoop() {
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stop_ && pending_.empty()) {
          cv_.wait(lock);
        }
        if (stop_) {
          return;
        }
      }

      PollPending();
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }

  void PollPending() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pending_.begin();
    while (it != pending_.end()) {
      musaError_t status = musaEventQuery(it->event);
      if (status == musaSuccess) {
        (void)musaEventDestroy(it->event);
        if (!CacheCompleted(*it)) {
          (void)musaFree(it->ptr);
        }
        it = pending_.erase(it);
      } else if (status == musaErrorNotReady) {
        ++it;
      } else {
        (void)musaEventDestroy(it->event);
        (void)musaFree(it->ptr);
        it = pending_.erase(it);
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::list<PendingFree> pending_;
  std::multimap<size_t, CachedBlock> cached_;
  size_t cached_bytes_ = 0;
  bool stop_ = false;
  std::thread worker_;
};

inline DeferredDeviceFreeQueue& GetDeferredDeviceFreeQueue() {
  static DeferredDeviceFreeQueue queue;
  return queue;
}

inline void FreeDeviceMemoryOnStream(void* ptr, musaStream_t stream) {
  GetDeferredDeviceFreeQueue().Free(ptr, stream);
}

inline void FreeDeviceMemoryOnStream(void* ptr, musaStream_t stream,
                                     size_t bytes) {
  GetDeferredDeviceFreeQueue().Free(ptr, stream, bytes);
}

inline void* AllocateDeviceMemoryOnStream(size_t bytes, musaStream_t stream) {
  return GetDeferredDeviceFreeQueue().Allocate(bytes, stream);
}

inline PinnedHostPool* GetKernelPinnedHostPool() {
  int device_id = 0;
  if (musaGetDevice(&device_id) != musaSuccess) {
    return nullptr;
  }

  // Keep kernel-side pinned pools alive until process exit. Destroying pinned
  // host pools during static shutdown can race MUSA runtime teardown.
  static auto* mutex = new std::mutex;
  static auto* pools =
      new std::unordered_map<int, std::unique_ptr<PinnedHostPool>>;
  std::lock_guard<std::mutex> lock(*mutex);
  auto it = pools->find(device_id);
  if (it != pools->end()) {
    return it->second.get();
  }

  auto pool = std::make_unique<PinnedHostPool>(device_id);
  PinnedHostPool* result = pool.get();
  pools->emplace(device_id, std::move(pool));
  return result;
}

inline OrtStatus* CopyTemporaryHostToDevice(
    void* dst, const void* src, size_t num_bytes, musaStream_t stream,
    bool wait_for_default_stream = false) {
  if (num_bytes == 0) {
    return nullptr;
  }

  if (stream == nullptr) {
    musaError_t status =
        musaMemcpy(dst, src, num_bytes, musaMemcpyHostToDevice);
    return status == musaSuccess ? nullptr
                                 : Ort::GetApi().CreateStatus(
                                       ORT_EP_FAIL, MusaErrorString(status));
  }

  if (wait_for_default_stream) {
    RETURN_IF_ERROR(WaitForDefaultStreamBeforeHostCopy(stream));
  }

  PinnedHostPool* pool = GetKernelPinnedHostPool();
  if (pool != nullptr) {
    void* staging = pool->Allocate(num_bytes);
    if (staging != nullptr) {
      std::memcpy(staging, src, num_bytes);
      musaError_t status = musaMemcpyAsync(dst, staging, num_bytes,
                                           musaMemcpyHostToDevice, stream);
      if (status != musaSuccess) {
        pool->FreeCompleted(staging);
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
      }

      pool->FreeAsync(staging, stream);
      return nullptr;
    }
  }

  musaError_t status =
      musaMemcpyAsync(dst, src, num_bytes, musaMemcpyHostToDevice, stream);
  if (status == musaSuccess) {
    // Without pinned staging, the host source may be a temporary vector owned
    // by the caller. Keep it alive until the pageable copy is no longer using
    // it.
    status = musaStreamSynchronize(stream);
  }
  return status == musaSuccess
             ? nullptr
             : Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
}
inline bool IsGpuMemory(const OrtMemoryInfo* memory_info) {
  const OrtMemoryDevice* device =
      Ort::GetEpApi().MemoryInfo_GetMemoryDevice(memory_info);
  return Ort::GetEpApi().MemoryDevice_GetDeviceType(device) ==
         OrtMemoryInfoDeviceType_GPU;
}

class DeviceInputBuffer {
 public:
  DeviceInputBuffer() = default;
  ~DeviceInputBuffer() {
    if (ptr_ != nullptr) {
      FreeDeviceMemoryOnStream(ptr_, stream_, bytes_);
    }
  }

  DeviceInputBuffer(const DeviceInputBuffer&) = delete;
  DeviceInputBuffer& operator=(const DeviceInputBuffer&) = delete;

  OrtStatus* Bind(Ort::ConstValue value, musaStream_t stream = nullptr) {
    if (IsGpuMemory(value.GetTensorMemoryInfo())) {
      data_ = value.GetTensorRawData();
      return nullptr;
    }

    bytes_ = value.GetTensorSizeInBytes();
    if (bytes_ == 0) {
      data_ = value.GetTensorRawData();
      return nullptr;
    }

    stream_ = stream;
    ptr_ = AllocateDeviceMemoryOnStream(bytes_, stream_);
    if (ptr_ == nullptr) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL, MusaErrorString(musaErrorMemoryAllocation));
    }
    RETURN_IF_ERROR(CopyTemporaryHostToDevice(ptr_, value.GetTensorRawData(),
                                              bytes_, stream_));
    data_ = ptr_;
    return nullptr;
  }

  const void* data() const { return data_; }

 private:
  void* ptr_ = nullptr;
  const void* data_ = nullptr;
  size_t bytes_ = 0;
  musaStream_t stream_ = nullptr;
};

inline OrtStatus* CopyToHost(Ort::ConstValue value, std::vector<uint8_t>& bytes,
                             musaStream_t stream = nullptr) {
  size_t num_bytes = value.GetTensorSizeInBytes();
  bytes.resize(num_bytes);
  if (num_bytes == 0) {
    return nullptr;
  }

  const void* src = value.GetTensorRawData();
  if (IsGpuMemory(value.GetTensorMemoryInfo())) {
    musaError_t status =
        stream != nullptr
            ? musaMemcpyAsync(bytes.data(), src, num_bytes,
                              musaMemcpyDeviceToHost, stream)
            : musaMemcpy(bytes.data(), src, num_bytes, musaMemcpyDeviceToHost);
    if (status == musaSuccess && stream != nullptr) {
      status = musaStreamSynchronize(stream);
    }
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
  } else {
    std::memcpy(bytes.data(), src, num_bytes);
  }

  return nullptr;
}

inline OrtStatus* CopyFromHost(Ort::UnownedValue value, const void* src,
                               size_t num_bytes,
                               musaStream_t stream = nullptr) {
  if (num_bytes == 0) {
    return nullptr;
  }

  void* dst = value.GetTensorMutableRawData();
  if (IsGpuMemory(value.GetTensorMemoryInfo())) {
    RETURN_IF_ERROR(CopyTemporaryHostToDevice(
        dst, src, num_bytes, stream, /*wait_for_default_stream*/ true));
  } else {
    std::memcpy(dst, src, num_bytes);
  }

  return nullptr;
}

inline OrtStatus* CopyRawTensor(Ort::ConstValue src_value,
                                Ort::UnownedValue dst_value, size_t num_bytes,
                                musaStream_t stream = nullptr) {
  if (num_bytes == 0) {
    return nullptr;
  }
  const void* src = src_value.GetTensorRawData();
  void* dst = dst_value.GetTensorMutableRawData();
  if (src == dst) {
    return nullptr;
  }

  const bool src_gpu = IsGpuMemory(src_value.GetTensorMemoryInfo());
  const bool dst_gpu = IsGpuMemory(dst_value.GetTensorMemoryInfo());
  if (src_gpu && dst_gpu) {
    musaError_t status =
        musaMemcpyAsync(dst, src, num_bytes, musaMemcpyDeviceToDevice, stream);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
  } else if (src_gpu) {
    musaError_t status =
        stream != nullptr
            ? musaMemcpyAsync(dst, src, num_bytes, musaMemcpyDeviceToHost,
                              stream)
            : musaMemcpy(dst, src, num_bytes, musaMemcpyDeviceToHost);
    if (status == musaSuccess && stream != nullptr) {
      status = musaStreamSynchronize(stream);
    }
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
  } else if (dst_gpu) {
    RETURN_IF_ERROR(CopyTemporaryHostToDevice(dst, src, num_bytes, stream));
  } else {
    std::memcpy(dst, src, num_bytes);
  }
  return nullptr;
}

inline OrtStatus* DeviceMemcpy(void* dst, const void* src, size_t num_bytes,
                               musaStream_t stream = nullptr) {
  if (num_bytes == 0 || dst == src) {
    return nullptr;
  }
  musaError_t status =
      musaMemcpyAsync(dst, src, num_bytes, musaMemcpyDeviceToDevice, stream);
  if (status != musaSuccess) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
  }
  return nullptr;
}

inline OrtStatus* DeviceMemcpy2D(void* dst, size_t dst_pitch, const void* src,
                                 size_t src_pitch, size_t width_bytes,
                                 size_t height, musaStream_t stream = nullptr) {
  if (width_bytes == 0 || height == 0) {
    return nullptr;
  }
  musaError_t status =
      musaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width_bytes, height,
                        musaMemcpyDeviceToDevice, stream);
  if (status != musaSuccess) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
  }
  return nullptr;
}

inline OrtStatus* LaunchStatus(musaError_t status) {
  if (status != musaSuccess) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
  }
  return nullptr;
}
inline bool AllGpuInputs(Ort::KernelContext& ctx) {
  for (size_t i = 0; i < ctx.GetInputCount(); ++i) {
    if (!IsGpuMemory(ctx.GetInput(i).GetTensorMemoryInfo())) {
      return false;
    }
  }
  return true;
}
