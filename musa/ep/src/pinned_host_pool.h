#pragma once

#include <musa_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <list>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

class PinnedHostPool {
 public:
  explicit PinnedHostPool(int device_id)
      : device_id_{device_id}, stop_polling_{false} {
    (void)musaSetDevice(device_id_);
    polling_thread_ = std::thread(&PinnedHostPool::PollLoop, this);
  }

  ~PinnedHostPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_polling_ = true;
      poll_cv_.notify_all();
    }

    if (polling_thread_.joinable()) {
      polling_thread_.join();
    }

    (void)musaSetDevice(device_id_);

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& block : pending_frees_) {
      if (block.event != nullptr) {
        while (musaEventQuery(block.event) == musaErrorNotReady) {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        (void)musaEventDestroy(block.event);
      }
      (void)musaFreeHost(block.ptr);
    }

    for (const auto& block : free_blocks_) {
      (void)musaFreeHost(block.second);
    }

    for (auto& item : live_blocks_) {
      (void)musaFreeHost(item.first);
    }

    for (auto event : reusable_events_) {
      (void)musaEventDestroy(event);
    }
  }

  void* Allocate(size_t requested_size) {
    if (requested_size == 0) {
      return nullptr;
    }

    const size_t size = RoundSize(requested_size);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      PollPendingFreesLocked();
      auto cached = free_blocks_.lower_bound(size);
      if (cached != free_blocks_.end()) {
        void* ptr = cached->second;
        const size_t block_size = cached->first;
        cached_bytes_ -= block_size;
        free_blocks_.erase(cached);
        live_blocks_[ptr] = block_size;
        return ptr;
      }
    }

    (void)musaSetDevice(device_id_);
    void* ptr = nullptr;
    if (musaHostAlloc(&ptr, size, musaHostAllocDefault) != musaSuccess) {
      return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    live_blocks_[ptr] = size;
    return ptr;
  }

  void FreeCompleted(void* ptr) {
    if (ptr == nullptr) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto live = live_blocks_.find(ptr);
    if (live == live_blocks_.end()) {
      (void)musaFreeHost(ptr);
      return;
    }

    const size_t size = live->second;
    live_blocks_.erase(live);
    if (ShouldReleaseBlockLocked(size)) {
      (void)musaFreeHost(ptr);
      return;
    }

    free_blocks_.emplace(size, ptr);
    cached_bytes_ += size;
  }

  void FreeAsync(void* ptr, musaStream_t stream) {
    if (ptr == nullptr) {
      return;
    }

    if (stream == nullptr) {
      FreeCompleted(ptr);
      return;
    }

    size_t size = 0;
    musaEvent_t event = nullptr;
    bool release_after_event = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto live = live_blocks_.find(ptr);
      if (live == live_blocks_.end()) {
        (void)musaFreeHost(ptr);
        return;
      }

      size = live->second;
      live_blocks_.erase(live);
      release_after_event = ShouldReleaseBlockLocked(size);

      if (!reusable_events_.empty()) {
        event = reusable_events_.back();
        reusable_events_.pop_back();
      }
    }

    (void)musaSetDevice(device_id_);
    if (event == nullptr &&
        musaEventCreateWithFlags(&event, musaEventDisableTiming) != musaSuccess) {
      (void)musaStreamSynchronize(stream);
      FreeCompleted(ptr);
      return;
    }

    if (musaEventRecord(event, stream) != musaSuccess) {
      (void)musaEventDestroy(event);
      (void)musaStreamSynchronize(stream);
      FreeCompleted(ptr);
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_frees_.push_back({ptr, size, event, release_after_event});
      poll_cv_.notify_one();
    }
  }

 private:
  struct PendingBlock {
    void* ptr = nullptr;
    size_t size = 0;
    musaEvent_t event = nullptr;
    bool release_after_event = false;
  };

  static size_t RoundSize(size_t size) {
    constexpr size_t kAlignment = 256;
    return (size + kAlignment - 1) & ~(kAlignment - 1);
  }

  static size_t CacheLimitBytes() {
    const char* env = std::getenv("ORT_MUSA_PINNED_POOL_CACHE_LIMIT_MB");
    if (env == nullptr || *env == '\0') {
      return size_t{1024} * 1024 * 1024;
    }

    long mb = std::strtol(env, nullptr, 10);
    return mb <= 0 ? 0 : static_cast<size_t>(mb) * 1024 * 1024;
  }

  bool ShouldReleaseBlockLocked(size_t size) const {
    const size_t limit = CacheLimitBytes();
    return limit == 0 || cached_bytes_ + size > limit;
  }

  void PollLoop() {
    (void)musaSetDevice(device_id_);
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stop_polling_ && pending_frees_.empty()) {
          poll_cv_.wait(lock);
        }

        PollPendingFreesLocked();
        if (stop_polling_) {
          return;
        }
      }

      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }

  void PollPendingFreesLocked() {
    auto it = pending_frees_.begin();
    while (it != pending_frees_.end()) {
      musaError_t status = musaEventQuery(it->event);
      if (status == musaSuccess) {
        reusable_events_.push_back(it->event);
        if (it->release_after_event) {
          (void)musaFreeHost(it->ptr);
        } else {
          cached_bytes_ += it->size;
          free_blocks_.emplace(it->size, it->ptr);
        }
        it = pending_frees_.erase(it);
      } else if (status == musaErrorNotReady) {
        ++it;
      } else {
        (void)musaEventDestroy(it->event);
        if (it->release_after_event) {
          (void)musaFreeHost(it->ptr);
        } else {
          cached_bytes_ += it->size;
          free_blocks_.emplace(it->size, it->ptr);
        }
        it = pending_frees_.erase(it);
      }
    }
  }

  const int device_id_;
  std::mutex mutex_;
  std::multimap<size_t, void*> free_blocks_;
  std::unordered_map<void*, size_t> live_blocks_;
  std::list<PendingBlock> pending_frees_;
  std::vector<musaEvent_t> reusable_events_;
  size_t cached_bytes_ = 0;
  std::condition_variable poll_cv_;
  std::atomic<bool> stop_polling_;
  std::thread polling_thread_;
};
