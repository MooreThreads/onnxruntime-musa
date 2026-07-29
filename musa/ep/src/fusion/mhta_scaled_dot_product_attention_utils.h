// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include <string>

namespace musa_ep {

// Einsum labels are arbitrary. These two forms describe the same rank-3
// attention contraction and are emitted by different export branches.
inline bool IsSupportedMhtaSimRank3Equation(const std::string& equation) {
  return equation == "ilhw,bjhw->bhl" || equation == "blhw,bjhw->bhl";
}

}  // namespace musa_ep
