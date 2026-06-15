// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>

#include <mutex>
#include <unordered_map>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "runtime/ep_musa_utils.h"

class PendingMusaWorkRegistry {
 public:
  ~PendingMusaWorkRegistry() {
    std::lock_guard<std::mutex> lock(mutex_);
    DestroyEventsLocked(pending_);
    DestroyEventsLocked(buffer_uses_);
    DestroySingleEventsLocked(ready_);
  }

  OrtStatus* Register(const void* ptr, musaStream_t producer_stream) {
    return RegisterEvent(ptr, producer_stream, pending_);
  }

  OrtStatus* RegisterBufferUse(const void* ptr, musaStream_t stream) {
    return RegisterEvent(ptr, stream, buffer_uses_);
  }

  OrtStatus* Wait(const void* ptr, musaStream_t consumer_stream) {
    if (ptr == nullptr) {
      return nullptr;
    }

    std::vector<musaEvent_t> events;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      CleanupCompletedLocked();
      CollectEventsForPtrLocked(pending_, ptr, events);
    }

    return WaitEvents(events, consumer_stream);
  }

  OrtStatus* WaitForBufferReuse(const void* ptr) {
    if (ptr == nullptr) {
      return nullptr;
    }

    std::vector<musaEvent_t> events;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      CleanupCompletedLocked();
      CollectEventsForPtrLocked(pending_, ptr, events);
      CollectEventsForPtrLocked(buffer_uses_, ptr, events);
    }

    OrtStatus* status = WaitEvents(events, nullptr);
    if (status == nullptr) {
      CleanupCompleted();
    }
    return status;
  }

  OrtStatus* EnsureReady(const void* ptr, musaStream_t producer_stream,
                         musaStream_t consumer_stream) {
    if (ptr == nullptr || producer_stream == nullptr ||
        consumer_stream == producer_stream) {
      return nullptr;
    }

    musaEvent_t event = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = ready_.find(ptr);
      if (it != ready_.end()) {
        event = it->second;
      }
    }

    if (event == nullptr) {
      musaError_t status =
          musaEventCreateWithFlags(&event, musaEventDisableTiming);
      if (status != musaSuccess) {
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
      }

      status = musaEventRecord(event, producer_stream);
      if (status != musaSuccess) {
        (void)musaEventDestroy(event);
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
      }

      std::lock_guard<std::mutex> lock(mutex_);
      auto existing = ready_.find(ptr);
      if (existing != ready_.end()) {
        (void)musaEventDestroy(event);
        event = existing->second;
      } else {
        ready_.emplace(ptr, event);
      }
    }

    musaError_t wait_status = musaStreamWaitEvent(consumer_stream, event, 0);
    return wait_status == musaSuccess
               ? nullptr
               : Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                             MusaErrorString(wait_status));
  }

  OrtStatus* MarkReady(const void* ptr, musaStream_t producer_stream) {
    if (ptr == nullptr || producer_stream == nullptr) {
      return nullptr;
    }

    musaEvent_t event = nullptr;
    musaError_t status =
        musaEventCreateWithFlags(&event, musaEventDisableTiming);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }

    status = musaEventRecord(event, producer_stream);
    if (status != musaSuccess) {
      (void)musaEventDestroy(event);
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto old = ready_.find(ptr);
    if (old != ready_.end()) {
      (void)musaEventDestroy(old->second);
      old->second = event;
    } else {
      ready_.emplace(ptr, event);
    }
    return nullptr;
  }

  bool HasPendingForBufferReuse(const void* ptr) {
    if (ptr == nullptr) {
      return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    CleanupCompletedLocked();
    return pending_.find(ptr) != pending_.end() ||
           buffer_uses_.find(ptr) != buffer_uses_.end();
  }

  bool HasPendingProducer(const void* ptr) {
    if (ptr == nullptr) {
      return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    CleanupCompletedLocked();
    return pending_.find(ptr) != pending_.end();
  }

  OrtStatus* WaitAll(musaStream_t consumer_stream) {
    std::vector<musaEvent_t> events;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      CleanupCompletedLocked();
      for (const auto& entry : pending_) {
        events.insert(events.end(), entry.second.begin(), entry.second.end());
      }
      for (const auto& entry : buffer_uses_) {
        events.insert(events.end(), entry.second.begin(), entry.second.end());
      }
    }

    OrtStatus* status = WaitEvents(events, consumer_stream);
    if (status == nullptr) {
      CleanupCompleted();
    }
    return status;
  }

  void CleanupCompleted() {
    std::lock_guard<std::mutex> lock(mutex_);
    CleanupCompletedLocked();
  }

  void Clear(const void* ptr) {
    if (ptr == nullptr) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    DestroyEventsForPtrLocked(pending_, ptr);
    DestroyEventsForPtrLocked(buffer_uses_, ptr);
    auto ready = ready_.find(ptr);
    if (ready != ready_.end()) {
      (void)musaEventDestroy(ready->second);
      ready_.erase(ready);
    }
  }

 private:
  using EventMap = std::unordered_map<const void*, std::vector<musaEvent_t>>;
  using SingleEventMap = std::unordered_map<const void*, musaEvent_t>;

  OrtStatus* RegisterEvent(const void* ptr, musaStream_t stream,
                           EventMap& event_map) {
    if (ptr == nullptr) {
      return nullptr;
    }

    musaEvent_t event = nullptr;
    musaError_t status =
        musaEventCreateWithFlags(&event, musaEventDisableTiming);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }

    status = musaEventRecord(event, stream);
    if (status != musaSuccess) {
      (void)musaEventDestroy(event);
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }

    std::lock_guard<std::mutex> lock(mutex_);
    CleanupCompletedLocked();
    event_map[ptr].push_back(event);
    return nullptr;
  }

  OrtStatus* WaitEvents(const std::vector<musaEvent_t>& events,
                        musaStream_t consumer_stream) {
    for (musaEvent_t event : events) {
      musaError_t status =
          consumer_stream != nullptr
              ? musaStreamWaitEvent(consumer_stream, event, 0)
              : musaEventSynchronize(event);
      if (status != musaSuccess) {
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
      }
    }
    return nullptr;
  }

  void CollectEventsForPtrLocked(const EventMap& event_map, const void* ptr,
                                 std::vector<musaEvent_t>& events) const {
    auto it = event_map.find(ptr);
    if (it == event_map.end()) {
      return;
    }
    events.insert(events.end(), it->second.begin(), it->second.end());
  }

  void DestroyEventsLocked(EventMap& event_map) {
    for (auto& entry : event_map) {
      for (musaEvent_t event : entry.second) {
        if (event != nullptr) {
          (void)musaEventDestroy(event);
        }
      }
    }
    event_map.clear();
  }

  void DestroyEventsForPtrLocked(EventMap& event_map, const void* ptr) {
    auto it = event_map.find(ptr);
    if (it == event_map.end()) {
      return;
    }
    for (musaEvent_t event : it->second) {
      if (event != nullptr) {
        (void)musaEventDestroy(event);
      }
    }
    event_map.erase(it);
  }

  void DestroySingleEventsLocked(SingleEventMap& event_map) {
    for (auto& entry : event_map) {
      if (entry.second != nullptr) {
        (void)musaEventDestroy(entry.second);
      }
    }
    event_map.clear();
  }

  void CleanupCompletedLocked() {
    CleanupCompletedLocked(pending_);
    CleanupCompletedLocked(buffer_uses_);
  }

  void CleanupCompletedLocked(EventMap& event_map) {
    for (auto map_it = event_map.begin(); map_it != event_map.end();) {
      std::vector<musaEvent_t>& events = map_it->second;
      for (auto event_it = events.begin(); event_it != events.end();) {
        musaError_t status = musaEventQuery(*event_it);
        if (status == musaSuccess || status != musaErrorNotReady) {
          (void)musaEventDestroy(*event_it);
          event_it = events.erase(event_it);
        } else {
          ++event_it;
        }
      }

      if (events.empty()) {
        map_it = event_map.erase(map_it);
      } else {
        ++map_it;
      }
    }
  }

  std::mutex mutex_;
  EventMap pending_;
  EventMap buffer_uses_;
  SingleEventMap ready_;
};

inline PendingMusaWorkRegistry& GetPendingMusaWorkRegistry() {
  static auto* registry = new PendingMusaWorkRegistry;
  return *registry;
}

inline OrtStatus* RegisterPendingMusaWork(const void* ptr,
                                          musaStream_t producer_stream) {
  return GetPendingMusaWorkRegistry().Register(ptr, producer_stream);
}

inline OrtStatus* WaitForPendingMusaWork(const void* ptr,
                                         musaStream_t consumer_stream) {
  return GetPendingMusaWorkRegistry().Wait(ptr, consumer_stream);
}

inline bool HasPendingMusaProducer(const void* ptr) {
  return GetPendingMusaWorkRegistry().HasPendingProducer(ptr);
}

inline OrtStatus* EnsureMusaBufferReadyOnStream(const void* ptr,
                                                musaStream_t producer_stream,
                                                musaStream_t consumer_stream) {
  return GetPendingMusaWorkRegistry().EnsureReady(ptr, producer_stream,
                                                  consumer_stream);
}

inline OrtStatus* MarkMusaBufferReady(const void* ptr,
                                      musaStream_t producer_stream) {
  return GetPendingMusaWorkRegistry().MarkReady(ptr, producer_stream);
}

inline OrtStatus* RegisterPendingMusaBufferUse(const void* ptr,
                                               musaStream_t stream) {
  return GetPendingMusaWorkRegistry().RegisterBufferUse(ptr, stream);
}

inline bool HasPendingMusaWorkForBufferReuse(const void* ptr) {
  return GetPendingMusaWorkRegistry().HasPendingForBufferReuse(ptr);
}

inline OrtStatus* WaitForPendingMusaWorkForBufferReuse(const void* ptr) {
  return GetPendingMusaWorkRegistry().WaitForBufferReuse(ptr);
}

inline void ClearPendingMusaBufferState(const void* ptr) {
  GetPendingMusaWorkRegistry().Clear(ptr);
}

inline OrtStatus* WaitForAllPendingMusaWork(musaStream_t consumer_stream) {
  return GetPendingMusaWorkRegistry().WaitAll(consumer_stream);
}

inline void CleanupCompletedPendingMusaWork() {
  GetPendingMusaWorkRegistry().CleanupCompleted();
}
