#pragma once

#include "plugin_ep_utils.h"
#include "pinned_host_pool.h"

#include <musa_runtime.h>

#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    AllocOnStream = AllocOnStreamImpl;
    Shrink = nullptr;
    RegisterAllocator(this);
  }

  ~CustomAllocator() override {
    UnregisterAllocator(this);
    for (auto& item : cached_blocks_) {
      (void)musaFree(item.second.ptr);
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
    return impl.AllocateCached(size, nullptr);
  }

  static void* ORT_API_CALL AllocOnStreamImpl(struct OrtAllocator* this_,
                                              size_t size,
                                              OrtSyncStream* stream) {
    if (size == 0) {
      return nullptr;
    }
    auto& impl = *static_cast<CustomAllocator*>(this_);
    const OrtSyncStreamImpl* stream_impl =
        stream != nullptr ? Ort::GetEpApi().SyncStream_GetImpl(stream)
                          : nullptr;
    return impl.AllocateCached(size, stream_impl);
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

  static void ResetBlocksUsingStream(const OrtSyncStreamImpl* stream_impl) {
    if (stream_impl == nullptr) {
      return;
    }

    std::vector<CustomAllocator*> allocators;
    {
      std::lock_guard<std::mutex> lock(RegistryMutex());
      allocators.assign(Registry().begin(), Registry().end());
    }

    for (CustomAllocator* allocator : allocators) {
      allocator->ResetBlocksUsingStreamImpl(stream_impl);
    }
  }

 private:
  struct BlockInfo {
    size_t size = 0;
    const OrtSyncStreamImpl* stream = nullptr;
  };

  struct CachedBlock {
    void* ptr = nullptr;
    const OrtSyncStreamImpl* stream = nullptr;
  };

  static std::mutex& RegistryMutex() {
    static auto* mutex = new std::mutex;
    return *mutex;
  }

  static std::unordered_set<CustomAllocator*>& Registry() {
    static auto* registry = new std::unordered_set<CustomAllocator*>;
    return *registry;
  }

  static void RegisterAllocator(CustomAllocator* allocator) {
    std::lock_guard<std::mutex> lock(RegistryMutex());
    Registry().insert(allocator);
  }

  static void UnregisterAllocator(CustomAllocator* allocator) {
    std::lock_guard<std::mutex> lock(RegistryMutex());
    Registry().erase(allocator);
  }

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

  void* AllocateCached(size_t requested_size,
                       const OrtSyncStreamImpl* stream) {
    const size_t size = RoundSize(requested_size);
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto cached = cached_blocks_.lower_bound(size);
         cached != cached_blocks_.end(); ++cached) {
      if (cached->second.stream != stream) {
        continue;
      }

      void* p = cached->second.ptr;
      const size_t block_size = cached->first;
      cached_bytes_ -= block_size;
      cached_blocks_.erase(cached);
      live_blocks_[p] = BlockInfo{block_size, stream};
      return p;
    }

    void* p = nullptr;
    musaError_t status = musaMalloc(&p, size);
    if (status != musaSuccess) {
      return nullptr;
    }
    live_blocks_[p] = BlockInfo{size, stream};
    return p;
  }

  void FreeCached(void* p) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto live = live_blocks_.find(p);
    if (live == live_blocks_.end()) {
      (void)musaFree(p);
      return;
    }

    const BlockInfo block = live->second;
    live_blocks_.erase(live);
    const size_t limit = CacheLimitBytes();
    if (limit == 0 || cached_bytes_ + block.size > limit) {
      (void)musaFree(p);
      return;
    }

    cached_blocks_.emplace(block.size, CachedBlock{p, block.stream});
    cached_bytes_ += block.size;
  }

  void ResetBlocksUsingStreamImpl(const OrtSyncStreamImpl* stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : cached_blocks_) {
      if (item.second.stream == stream) {
        item.second.stream = nullptr;
      }
    }
    for (auto& item : live_blocks_) {
      if (item.second.stream == stream) {
        item.second.stream = nullptr;
      }
    }
  }

  const OrtMemoryInfo* memory_info;
  std::mutex mutex_;
  std::multimap<size_t, CachedBlock> cached_blocks_;
  std::unordered_map<void*, BlockInfo> live_blocks_;
  size_t cached_bytes_ = 0;
};

struct PinnedHostAllocator : BaseAllocator {
  PinnedHostAllocator(const OrtMemoryInfo* mem_info,
                      std::shared_ptr<PinnedHostPool> pool)
      : memory_info{mem_info}, pool_{std::move(pool)} {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;
    GetStats = nullptr;
    AllocOnStream = nullptr;
    Shrink = nullptr;
  }

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size) {
    if (size == 0) {
      return nullptr;
    }

    auto& impl = *static_cast<PinnedHostAllocator*>(this_);
    return impl.pool_->Allocate(size);
  }

  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p) {
    if (p == nullptr) {
      return;
    }

    auto& impl = *static_cast<PinnedHostAllocator*>(this_);
    impl.pool_->FreeCompleted(p);
  }

  static const struct OrtMemoryInfo* ORT_API_CALL
  InfoImpl(const struct OrtAllocator* this_) {
    const auto& impl = *static_cast<const PinnedHostAllocator*>(this_);
    return impl.memory_info;
  }

 private:
  const OrtMemoryInfo* memory_info;
  std::shared_ptr<PinnedHostPool> pool_;
};

using AllocationUniquePtr = std::unique_ptr<void, std::function<void(void*)>>;

inline AllocationUniquePtr AllocateBytes(OrtAllocator* allocator, size_t num_bytes) {
  void* p = allocator->Alloc(allocator, num_bytes);
  return AllocationUniquePtr(p, [allocator](void* d) { allocator->Free(allocator, d); });
}
