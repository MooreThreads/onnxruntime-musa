#pragma once

#include <string>

namespace ort_musa {

struct MusaRuntimeInfo {
  bool available = false;
  int device_count = 0;
  std::string description;
};

MusaRuntimeInfo QueryMusaRuntime();

}  // namespace ort_musa
