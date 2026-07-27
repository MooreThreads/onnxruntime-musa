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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "shared_inc/device_kernel_types.h"

inline int64_t NumElements(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 1;
  }
  int64_t n = 1;
  for (int64_t dim : shape) {
    n *= dim;
  }
  return n;
}

inline int64_t NormalizeAxis(int64_t axis, size_t rank) {
  int64_t r = static_cast<int64_t>(rank);
  return axis < 0 ? axis + r : axis;
}

inline std::vector<int64_t> Strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

inline std::vector<int64_t> Coordinates(int64_t linear,
                                        const std::vector<int64_t>& shape) {
  std::vector<int64_t> coord(shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    int64_t dim = shape[static_cast<size_t>(i)];
    coord[static_cast<size_t>(i)] = dim == 0 ? 0 : linear % dim;
    linear = dim == 0 ? 0 : linear / dim;
  }
  return coord;
}

inline int64_t Offset(const std::vector<int64_t>& coord,
                      const std::vector<int64_t>& strides) {
  int64_t off = 0;
  for (size_t i = 0; i < coord.size(); ++i) {
    off += coord[i] * strides[i];
  }
  return off;
}

inline void AppendShapeForError(std::string& message,
                                const std::vector<int64_t>& shape) {
  message.push_back('[');
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      message.push_back(',');
    }
    message += std::to_string(shape[i]);
  }
  message.push_back(']');
}

inline std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& a,
                                           const std::vector<int64_t>& b) {
  size_t rank = std::max(a.size(), b.size());
  std::vector<int64_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (da != db && da != 1 && db != 1) {
      std::string message = "broadcast shape mismatch: lhs=";
      AppendShapeForError(message, a);
      message += " rhs=";
      AppendShapeForError(message, b);
      throw std::runtime_error(message);
    }
    out[i] = da == 0 || db == 0 ? 0 : std::max(da, db);
  }
  return out;
}

inline MusaBroadcastParams MakeBroadcastParams(
    const std::vector<int64_t>& out_shape,
    const std::vector<int64_t>& lhs_shape,
    const std::vector<int64_t>& rhs_shape) {
  MusaBroadcastParams params{};
  const size_t rank = out_shape.size();
  params.rank = static_cast<int32_t>(rank);
  params.total_elements = NumElements(out_shape);

  auto out_strides = Strides(out_shape);
  auto lhs_strides = Strides(lhs_shape);
  auto rhs_strides = Strides(rhs_shape);
  const size_t lhs_rank = lhs_shape.size();
  const size_t rhs_rank = rhs_shape.size();
  const size_t lhs_offset = rank - lhs_rank;
  const size_t rhs_offset = rank - rhs_rank;

  for (size_t dim = 0; dim < rank; ++dim) {
    params.output_strides[dim] = out_strides[dim];

    if (dim < lhs_offset) {
      params.lhs_strides[dim] = 0;
    } else {
      const size_t lhs_dim = dim - lhs_offset;
      params.lhs_strides[dim] =
          lhs_shape[lhs_dim] == 1 ? 0 : lhs_strides[lhs_dim];
    }

    if (dim < rhs_offset) {
      params.rhs_strides[dim] = 0;
    } else {
      const size_t rhs_dim = dim - rhs_offset;
      params.rhs_strides[dim] =
          rhs_shape[rhs_dim] == 1 ? 0 : rhs_strides[rhs_dim];
    }
  }

  return params;
}

inline bool CanUseBroadcastKernel(const std::vector<int64_t>& out_shape,
                                  const std::vector<int64_t>& lhs_shape,
                                  const std::vector<int64_t>& rhs_shape) {
  return out_shape.size() <= kMusaMaxBroadcastRank &&
         lhs_shape.size() <= kMusaMaxBroadcastRank &&
         rhs_shape.size() <= kMusaMaxBroadcastRank;
}

inline int64_t BroadcastOffset(const std::vector<int64_t>& out_coord,
                               const std::vector<int64_t>& in_shape,
                               const std::vector<int64_t>& in_strides) {
  size_t rank = out_coord.size();
  size_t in_rank = in_shape.size();
  int64_t off = 0;
  for (size_t i = 0; i < in_rank; ++i) {
    size_t out_i = rank - in_rank + i;
    int64_t c = in_shape[i] == 1 ? 0 : out_coord[out_i];
    off += c * in_strides[i];
  }
  return off;
}

inline std::vector<int64_t> PrefixShape(const std::vector<int64_t>& shape,
                                        size_t trailing_dims) {
  if (shape.size() < trailing_dims) {
    return {};
  }
  return std::vector<int64_t>(
      shape.begin(), shape.end() - static_cast<int64_t>(trailing_dims));
}

inline std::vector<int64_t> BroadcastBatchCoord(
    const std::vector<int64_t>& out_coord,
    const std::vector<int64_t>& out_shape,
    const std::vector<int64_t>& in_shape) {
  std::vector<int64_t> coord(in_shape.size(), 0);
  size_t out_rank = out_shape.size();
  size_t in_rank = in_shape.size();
  for (size_t i = 0; i < in_rank; ++i) {
    size_t out_i = out_rank - in_rank + i;
    coord[i] = in_shape[i] == 1 ? 0 : out_coord[out_i];
  }
  return coord;
}

inline std::set<int64_t> AxesSet(std::vector<int64_t> axes, size_t rank) {
  std::set<int64_t> out;
  if (axes.empty()) {
    for (size_t i = 0; i < rank; ++i) out.insert(static_cast<int64_t>(i));
    return out;
  }
  for (int64_t axis : axes) {
    out.insert(NormalizeAxis(axis, rank));
  }
  return out;
}
