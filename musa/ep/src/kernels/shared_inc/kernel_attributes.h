// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <vector>

#include "utils.h"

template <typename T>
T AttrOrDefault(Ort::ConstKernelInfo& info, const char* name, T default_value) {
  try {
    return info.GetAttribute<T>(name);
  } catch (...) {
    return default_value;
  }
}

inline std::vector<int64_t> AttrsOrEmpty(Ort::ConstKernelInfo& info,
                                         const char* name) {
  try {
    return info.GetAttributes<int64_t>(name);
  } catch (...) {
    return {};
  }
}
