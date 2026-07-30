// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "arena_ownership_index.h"

namespace {

using onnxruntime::musa_plugin::ArenaOwnershipIndex;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

const void* Key(std::uintptr_t value) {
  return reinterpret_cast<const void*>(value);
}

void TestAssignAndClear() {
  ArenaOwnershipIndex index;
  index.Assign(7, nullptr, nullptr, Key(1), Key(101));

  Check(index.FindStream(Key(101)) == Key(1), "implementation lookup failed");
  Check(index.FindChunks(Key(1))->count(7) == 1, "owner set is missing chunk");

  index.Assign(7, Key(1), Key(101), nullptr, nullptr);
  Check(index.FindStream(Key(101)) == nullptr,
        "empty implementation entry was not removed");
  Check(index.FindChunks(Key(1)) == nullptr,
        "empty stream entry was not removed");
}

void TestTransferRemovesOldOwner() {
  ArenaOwnershipIndex index;
  index.Assign(11, nullptr, nullptr, Key(1), Key(101));
  index.Assign(11, Key(1), Key(101), Key(2), Key(102));

  Check(index.FindChunks(Key(1)) == nullptr,
        "transfer retained the old owner set");
  Check(index.FindStream(Key(101)) == nullptr,
        "transfer retained the old implementation entry");
  Check(index.FindChunks(Key(2))->count(11) == 1,
        "transfer did not install the new owner");
  Check(index.FindStream(Key(102)) == Key(2),
        "transfer did not install the new implementation entry");
}

void TestTransferPreservesOtherOldOwnerChunks() {
  ArenaOwnershipIndex index;
  index.Assign(3, nullptr, nullptr, Key(1), Key(101));
  index.Assign(5, nullptr, nullptr, Key(1), Key(101));
  index.Assign(3, Key(1), Key(101), Key(2), Key(102));

  const auto* old_chunks = index.FindChunks(Key(1));
  Check(old_chunks != nullptr && old_chunks->size() == 1 &&
            old_chunks->count(5) == 1,
        "transfer damaged another chunk owned by the producer");
  Check(index.FindStream(Key(101)) == Key(1),
        "producer implementation entry was removed too early");
}

void TestIdempotentAssignment() {
  ArenaOwnershipIndex index;
  index.Assign(9, nullptr, nullptr, Key(1), Key(101));
  index.Assign(9, Key(1), Key(101), Key(1), Key(101));

  const auto* chunks = index.FindChunks(Key(1));
  Check(chunks != nullptr && chunks->size() == 1 && chunks->count(9) == 1,
        "idempotent assignment duplicated or removed a chunk");
}

void TestWrongOwnerIsRejected() {
  ArenaOwnershipIndex index;
  index.Assign(13, nullptr, nullptr, Key(1), Key(101));

  bool rejected = false;
  try {
    index.Assign(13, Key(2), Key(102), nullptr, nullptr);
  } catch (const std::logic_error&) {
    rejected = true;
  }
  Check(rejected, "clearing a chunk through the wrong owner was accepted");
  Check(index.FindChunks(Key(1))->count(13) == 1,
        "failed operation damaged the real owner");
}

void TestSharedImplementationIsRejected() {
  ArenaOwnershipIndex index;
  index.Assign(17, nullptr, nullptr, Key(1), Key(101));

  bool rejected = false;
  try {
    index.Assign(19, nullptr, nullptr, Key(2), Key(101));
  } catch (const std::logic_error&) {
    rejected = true;
  }
  Check(rejected, "one implementation was assigned to multiple streams");
}

void TestNullImplementationIsRejectedWithoutMutation() {
  ArenaOwnershipIndex index;

  bool rejected = false;
  try {
    index.Assign(23, nullptr, nullptr, Key(1), nullptr);
  } catch (const std::logic_error&) {
    rejected = true;
  }
  Check(rejected, "a stream without an implementation was accepted");
  Check(index.stream_to_chunks().empty() && index.impl_to_stream().empty(),
        "failed assignment mutated the index");
}

void TestClearOneChunkPreservesOwnerLookup() {
  ArenaOwnershipIndex index;
  index.Assign(29, nullptr, nullptr, Key(1), Key(101));
  index.Assign(31, nullptr, nullptr, Key(1), Key(101));
  index.Assign(29, Key(1), Key(101), nullptr, nullptr);

  Check(index.FindStream(Key(101)) == Key(1),
        "owner lookup was removed while another chunk remained");
  const auto* chunks = index.FindChunks(Key(1));
  Check(chunks != nullptr && chunks->size() == 1 && chunks->count(31) == 1,
        "clearing one chunk damaged its sibling");
}

void TestProducerReleaseAfterTransferPreservesConsumer() {
  ArenaOwnershipIndex index;
  index.Assign(37, nullptr, nullptr, Key(1), Key(101));
  index.Assign(37, Key(1), Key(101), Key(2), Key(102));

  Check(index.FindStream(Key(101)) == nullptr,
        "producer still has chunks to release after transfer");
  Check(index.FindStream(Key(102)) == Key(2) &&
            index.FindChunks(Key(2))->count(37) == 1,
        "producer release would clear the consumer's chunk");
}

void TestSplitRemainderTracksSameOwner() {
  ArenaOwnershipIndex index;
  index.Assign(41, nullptr, nullptr, Key(1), Key(101));
  index.Assign(43, nullptr, nullptr, Key(1), Key(101));

  const auto* chunks = index.FindChunks(Key(1));
  Check(chunks != nullptr && chunks->size() == 2 && chunks->count(41) == 1 &&
            chunks->count(43) == 1,
        "split remainder was not indexed with the original owner");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "expected one test case name\n";
    return EXIT_FAILURE;
  }

  const std::string_view test_name(argv[1]);
  if (test_name == "assign_and_clear") {
    TestAssignAndClear();
  } else if (test_name == "transfer_removes_old_owner") {
    TestTransferRemovesOldOwner();
  } else if (test_name == "transfer_preserves_other_chunks") {
    TestTransferPreservesOtherOldOwnerChunks();
  } else if (test_name == "idempotent_assignment") {
    TestIdempotentAssignment();
  } else if (test_name == "wrong_owner_rejected") {
    TestWrongOwnerIsRejected();
  } else if (test_name == "shared_implementation_rejected") {
    TestSharedImplementationIsRejected();
  } else if (test_name == "null_implementation_rejected") {
    TestNullImplementationIsRejectedWithoutMutation();
  } else if (test_name == "clear_one_preserves_lookup") {
    TestClearOneChunkPreservesOwnerLookup();
  } else if (test_name == "producer_release_after_transfer") {
    TestProducerReleaseAfterTransferPreservesConsumer();
  } else if (test_name == "split_remainder_tracks_owner") {
    TestSplitRemainderTracksSameOwner();
  } else {
    std::cerr << "unknown test case: " << test_name << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
