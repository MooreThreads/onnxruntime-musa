// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include <cstddef>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace onnxruntime {
namespace musa_plugin {

// Maintains the bidirectional ownership index used by the stream-ordered
// arena. A chunk is present in at most one stream set. Ownership changes must
// use Assign so removing the old owner and installing the new owner happen as
// one operation while the arena lock is held.
class ArenaOwnershipIndex {
 public:
  using ChunkHandle = size_t;
  using StreamKey = const void*;
  using StreamImplKey = const void*;
  using ChunkSet = std::set<ChunkHandle>;
  using StreamToChunks = std::unordered_map<StreamKey, ChunkSet>;
  using ImplToStream = std::unordered_map<StreamImplKey, StreamKey>;

  void Assign(ChunkHandle handle, StreamKey old_stream,
              StreamImplKey old_stream_impl, StreamKey new_stream,
              StreamImplKey new_stream_impl) {
    if (new_stream != nullptr && new_stream_impl == nullptr) {
      throw std::logic_error("stream implementation must not be null");
    }
    auto new_impl_it = impl_to_stream_.find(new_stream_impl);
    if (new_stream != nullptr && new_impl_it != impl_to_stream_.end() &&
        new_impl_it->second != new_stream) {
      throw std::logic_error("stream implementation has multiple owners");
    }

    if (old_stream != new_stream && old_stream != nullptr) {
      auto old_it = stream_to_chunks_.find(old_stream);
      if (old_it == stream_to_chunks_.end() ||
          old_it->second.erase(handle) != 1) {
        throw std::logic_error("chunk is missing from its owner set");
      }

      if (old_it->second.empty()) {
        auto impl_it = impl_to_stream_.find(old_stream_impl);
        if (impl_it != impl_to_stream_.end() && impl_it->second == old_stream) {
          impl_to_stream_.erase(impl_it);
        }
        stream_to_chunks_.erase(old_it);
      }
    }

    if (new_stream == nullptr) {
      return;
    }

    stream_to_chunks_[new_stream].insert(handle);
    impl_to_stream_[new_stream_impl] = new_stream;
  }

  StreamKey FindStream(StreamImplKey stream_impl) const {
    auto it = impl_to_stream_.find(stream_impl);
    return it == impl_to_stream_.end() ? nullptr : it->second;
  }

  const ChunkSet* FindChunks(StreamKey stream) const {
    auto it = stream_to_chunks_.find(stream);
    return it == stream_to_chunks_.end() ? nullptr : &it->second;
  }

  const StreamToChunks& stream_to_chunks() const { return stream_to_chunks_; }
  const ImplToStream& impl_to_stream() const { return impl_to_stream_; }

 private:
  StreamToChunks stream_to_chunks_;
  ImplToStream impl_to_stream_;
};

}  // namespace musa_plugin
}  // namespace onnxruntime
