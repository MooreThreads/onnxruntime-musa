// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>

#include "onnxruntime_c_api.h"

namespace musa_ep {

inline constexpr char kGeluDefaultApproximate[] = "none";

/// The ONNX Gelu (opset 20) `approximate` attribute: this MUSA route
/// implements the exact erf form only. The graph-level capability predicate
/// and the kernel's runtime guard both ask this one function, so the nodes
/// this route claims are the nodes it runs.
inline const char* GeluApproximateRejection(const std::string& approximate) {
  if (approximate == kGeluDefaultApproximate) {
    return nullptr;
  }
  return "Gelu only supports approximate=none in this MUSA route";
}

}  // namespace musa_ep
