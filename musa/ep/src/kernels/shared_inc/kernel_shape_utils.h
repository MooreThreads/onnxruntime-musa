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
  params.total_elements = NumElements(out_shape);

  auto out_strides = Strides(out_shape);
  auto lhs_strides = Strides(lhs_shape);
  auto rhs_strides = Strides(rhs_shape);
  const size_t lhs_rank = lhs_shape.size();
  const size_t rhs_rank = rhs_shape.size();
  const size_t lhs_offset = rank - lhs_rank;
  const size_t rhs_offset = rank - rhs_rank;

  std::vector<int64_t> dims;
  std::vector<int64_t> compressed_out_strides;
  std::vector<int64_t> compressed_lhs_strides;
  std::vector<int64_t> compressed_rhs_strides;
  dims.reserve(rank);
  compressed_out_strides.reserve(rank);
  compressed_lhs_strides.reserve(rank);
  compressed_rhs_strides.reserve(rank);

  const auto can_merge_stride = [](int64_t outer, int64_t inner,
                                   int64_t inner_dim) {
    return (outer == 0 && inner == 0) || outer == inner_dim * inner;
  };

  for (size_t dim = 0; dim < rank; ++dim) {
    const int64_t lhs_stride = dim < lhs_offset
                                   ? 0
                                   : (lhs_shape[dim - lhs_offset] == 1
                                          ? 0
                                          : lhs_strides[dim - lhs_offset]);
    const int64_t rhs_stride = dim < rhs_offset
                                   ? 0
                                   : (rhs_shape[dim - rhs_offset] == 1
                                          ? 0
                                          : rhs_strides[dim - rhs_offset]);

    // Leading output dimensions of size one never contribute to an index.
    // Dropping them reduces the number of divisions in ResolveBroadcastIndices
    // without changing either operand offset.
    if (dims.empty() && out_shape[dim] == 1) {
      continue;
    }

    const bool can_merge =
        !dims.empty() &&
        compressed_out_strides.back() == out_shape[dim] * out_strides[dim] &&
        can_merge_stride(compressed_lhs_strides.back(), lhs_stride,
                         out_shape[dim]) &&
        can_merge_stride(compressed_rhs_strides.back(), rhs_stride,
                         out_shape[dim]);
    if (can_merge) {
      dims.back() *= out_shape[dim];
      compressed_out_strides.back() = out_strides[dim];
      compressed_lhs_strides.back() = lhs_stride;
      compressed_rhs_strides.back() = rhs_stride;
    } else {
      dims.push_back(out_shape[dim]);
      compressed_out_strides.push_back(out_strides[dim]);
      compressed_lhs_strides.push_back(lhs_stride);
      compressed_rhs_strides.push_back(rhs_stride);
    }
  }

  params.rank = static_cast<int32_t>(dims.size());
  for (size_t dim = 0; dim < dims.size(); ++dim) {
    params.output_strides[dim] = compressed_out_strides[dim];
    params.lhs_strides[dim] = compressed_lhs_strides[dim];
    params.rhs_strides[dim] = compressed_rhs_strides[dim];
  }

  return params;
}

// Classify the bias layouts produced by Gemm/FusedGemm. These modes let the
// post kernel use direct indexing for the common [N], [1,N], and full output
// shapes while retaining the generic broadcast fallback for all other valid
// ONNX broadcasts.
inline MusaGemmPostBroadcast ClassifyGemmPostBroadcast(
    const std::vector<int64_t>& out_shape,
    const std::vector<int64_t>& bias_shape, int64_t& inner_size) {
  inner_size = 0;
  if (bias_shape.empty() || out_shape.empty()) {
    return MusaGemmPostBroadcast::Generic;
  }

  const int64_t last_dim = out_shape.back();
  if (last_dim <= 0) {
    return MusaGemmPostBroadcast::Generic;
  }
  if (NumElements(bias_shape) == 1) {
    return MusaGemmPostBroadcast::Scalar;
  }
  if ((bias_shape.size() == 1 && bias_shape[0] == last_dim) ||
      (bias_shape.size() == 2 && bias_shape[0] == 1 &&
       bias_shape[1] == last_dim)) {
    inner_size = last_dim;
    return MusaGemmPostBroadcast::LastDim;
  }
  if (bias_shape == out_shape) {
    return MusaGemmPostBroadcast::Full;
  }
  return MusaGemmPostBroadcast::Generic;
}

struct MusaMudnnBroadcastShapes {
  std::vector<int64_t> output;
  std::vector<int64_t> lhs;
  std::vector<int64_t> rhs;
};

// Collapse only dimensions whose flattened addressing is valid for both
// operands. The returned shapes describe the same byte ranges as the
// originals, but can be passed to muDNN when the original rank is too large.
inline MusaMudnnBroadcastShapes CompressBroadcastShapesForMudnn(
    const std::vector<int64_t>& out_shape,
    const std::vector<int64_t>& lhs_shape,
    const std::vector<int64_t>& rhs_shape) {
  const size_t rank = out_shape.size();
  const auto out_strides = Strides(out_shape);
  const auto lhs_strides = Strides(lhs_shape);
  const auto rhs_strides = Strides(rhs_shape);
  const size_t lhs_offset = rank - lhs_shape.size();
  const size_t rhs_offset = rank - rhs_shape.size();

  MusaMudnnBroadcastShapes result;
  result.output.reserve(rank);
  result.lhs.reserve(rank);
  result.rhs.reserve(rank);
  std::vector<int64_t> output_strides;
  std::vector<int64_t> lhs_compressed_strides;
  std::vector<int64_t> rhs_compressed_strides;
  output_strides.reserve(rank);
  lhs_compressed_strides.reserve(rank);
  rhs_compressed_strides.reserve(rank);

  const auto can_merge_stride = [](int64_t outer, int64_t inner,
                                   int64_t inner_dim) {
    return (outer == 0 && inner == 0) || outer == inner_dim * inner;
  };

  for (size_t dim = 0; dim < rank; ++dim) {
    const int64_t lhs_dim = dim < lhs_offset ? 1 : lhs_shape[dim - lhs_offset];
    const int64_t rhs_dim = dim < rhs_offset ? 1 : rhs_shape[dim - rhs_offset];
    const int64_t lhs_stride = lhs_dim == 1 ? 0 : lhs_strides[dim - lhs_offset];
    const int64_t rhs_stride = rhs_dim == 1 ? 0 : rhs_strides[dim - rhs_offset];

    if (result.output.empty() && out_shape[dim] == 1) continue;

    const bool can_merge =
        !result.output.empty() &&
        output_strides.back() == out_shape[dim] * out_strides[dim] &&
        can_merge_stride(lhs_compressed_strides.back(), lhs_stride,
                         out_shape[dim]) &&
        can_merge_stride(rhs_compressed_strides.back(), rhs_stride,
                         out_shape[dim]);
    if (can_merge) {
      result.output.back() *= out_shape[dim];
      result.lhs.back() *= lhs_dim;
      result.rhs.back() *= rhs_dim;
      output_strides.back() = out_strides[dim];
      lhs_compressed_strides.back() = lhs_stride;
      rhs_compressed_strides.back() = rhs_stride;
    } else {
      result.output.push_back(out_shape[dim]);
      result.lhs.push_back(lhs_dim);
      result.rhs.push_back(rhs_dim);
      output_strides.push_back(out_strides[dim]);
      lhs_compressed_strides.push_back(lhs_stride);
      rhs_compressed_strides.push_back(rhs_stride);
    }
  }
  return result;
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
