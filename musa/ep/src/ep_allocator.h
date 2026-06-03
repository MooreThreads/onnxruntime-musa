#pragma once

#include "plugin_ep_utils.h"

#include <musa_runtime.h>

#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>

struct BaseAllocator : OrtAllocator {
  virtual ~BaseAllocator() = default;
};

using AllocatorUniquePtr = std::unique_ptr<BaseAllocator>;

struct CustomAllocator : BaseAllocator {
  explicit CustomAllocator(const OrtMemoryInfo* mem_info) : memory_info{mem_info} {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;
    GetStats = nullptr;
    AllocOnStream = nullptr;
    Shrink = nullptr;
  }

  ~CustomAllocator() override {
    for (auto& item : cached_blocks_) {
      (void)musaFree(item.second);
    }
    for (auto& item : live_blocks_) {
      (void)musaFree(item.first);
    }
  }

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size) {
    if (size == 0) {
      return nullptr;
    }
    auto& impl = *static_cast<CustomAllocator*>(this_);
    return impl.AllocateCached(size);
  }

  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p) {
    if (p == nullptr) {
      return;
    }
    auto& impl = *static_cast<CustomAllocator*>(this_);
    impl.FreeCached(p);
  }

  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const CustomAllocator& impl = *static_cast<const CustomAllocator*>(this_);
    return impl.memory_info;
  }

 private:
  static size_t RoundSize(size_t size) {
    constexpr size_t kAlignment = 256;
    return (size + kAlignment - 1) & ~(kAlignment - 1);
  }

  static size_t CacheLimitBytes() {
    const char* env = std::getenv("ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB");
    if (env == nullptr || *env == '\0') {
      return size_t{2048} * 1024 * 1024;
    }
    long mb = std::strtol(env, nullptr, 10);
    return mb <= 0 ? 0 : static_cast<size_t>(mb) * 1024 * 1024;
  }

  void* AllocateCached(size_t requested_size) {
    const size_t size = RoundSize(requested_size);
    std::lock_guard<std::mutex> lock(mutex_);
    auto cached = cached_blocks_.lower_bound(size);
    if (cached != cached_blocks_.end()) {
      void* p = cached->second;
      const size_t block_size = cached->first;
      cached_bytes_ -= block_size;
      cached_blocks_.erase(cached);
      live_blocks_[p] = block_size;
      return p;
    }

    void* p = nullptr;
    musaError_t status = musaMalloc(&p, size);
    if (status != musaSuccess) {
      return nullptr;
    }
    live_blocks_[p] = size;
    return p;
  }

  void FreeCached(void* p) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto live = live_blocks_.find(p);
    if (live == live_blocks_.end()) {
      (void)musaFree(p);
      return;
    }

    const size_t size = live->second;
    live_blocks_.erase(live);
    const size_t limit = CacheLimitBytes();
    if (limit == 0 || cached_bytes_ + size > limit) {
      (void)musaFree(p);
      return;
    }

    cached_blocks_.emplace(size, p);
    cached_bytes_ += size;
  }

  const OrtMemoryInfo* memory_info;
  std::mutex mutex_;
  std::multimap<size_t, void*> cached_blocks_;
  std::unordered_map<void*, size_t> live_blocks_;
  size_t cached_bytes_ = 0;
};

using AllocationUniquePtr = std::unique_ptr<void, std::function<void(void*)>>;

inline AllocationUniquePtr AllocateBytes(OrtAllocator* allocator, size_t num_bytes) {
  void* p = allocator->Alloc(allocator, num_bytes);
  return AllocationUniquePtr(p, [allocator](void* d) { allocator->Free(allocator, d); });
}
