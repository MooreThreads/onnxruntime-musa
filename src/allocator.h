#pragma once

#include <cstdlib>

#include "plugin_ep_utils.h"

struct MusaHostAllocator : OrtAllocator {
  explicit MusaHostAllocator(const OrtMemoryInfo* memory_info_in)
      : memory_info(memory_info_in) {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;
    GetStats = nullptr;
    AllocOnStream = nullptr;
    Shrink = nullptr;
  }

  static void* ORT_API_CALL AllocImpl(OrtAllocator*, size_t size) {
    return std::malloc(size);
  }

  static void ORT_API_CALL FreeImpl(OrtAllocator*, void* ptr) {
    std::free(ptr);
  }

  static const OrtMemoryInfo* ORT_API_CALL
  InfoImpl(const OrtAllocator* allocator) {
    const auto* self = static_cast<const MusaHostAllocator*>(allocator);
    return self->memory_info;
  }

  const OrtMemoryInfo* memory_info;
};
