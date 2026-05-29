#pragma once

#include "plugin_ep_utils.h"

#include <musa_runtime.h>

#include <functional>
#include <memory>

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

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* /*this_*/, size_t size) {
    if (size == 0) {
      return nullptr;
    }
    void* p = nullptr;
    musaError_t status = musaMalloc(&p, size);
    return status == musaSuccess ? p : nullptr;
  }

  static void ORT_API_CALL FreeImpl(struct OrtAllocator* /*this_*/, void* p) {
    if (p != nullptr) {
      (void)musaFree(p);
    }
  }

  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const CustomAllocator& impl = *static_cast<const CustomAllocator*>(this_);
    return impl.memory_info;
  }

 private:
  const OrtMemoryInfo* memory_info;
};

using AllocationUniquePtr = std::unique_ptr<void, std::function<void(void*)>>;

inline AllocationUniquePtr AllocateBytes(OrtAllocator* allocator, size_t num_bytes) {
  void* p = allocator->Alloc(allocator, num_bytes);
  return AllocationUniquePtr(p, [allocator](void* d) { allocator->Free(allocator, d); });
}
