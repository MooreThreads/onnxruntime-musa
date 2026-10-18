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

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace musa_ep {

// A small interpreter for the shape-arithmetic chains that produce Tile repeats
// and MHTA Slice ends. Such a chain only reads tensor SHAPES and constants, so
// a fused node can recompute the value from the shapes of its inputs instead of
// consuming the tensor that ORT copied to the device (which would otherwise
// require a device-to-host read and a compute-stream synchronize).
//
// The expression is built at fusion-creation time and is read-only afterwards,
// so evaluation is thread-safe.
struct ShapeExprRef {
  bool is_node = false;  // node result vs source tensor shape
  int index = 0;         // node index or source index
};

struct ShapeExprNode {
  std::string op;
  std::vector<ShapeExprRef> inputs;
  std::vector<int64_t> attrs;     // op-specific constants (axes, etc.)
  std::vector<int64_t> constant;  // for kConstant
};

struct ShapeExpr {
  std::vector<ShapeExprNode> nodes;
  std::vector<size_t> source_input_indices;  // fused-input index per source
  int root = -1;

  bool Empty() const { return root < 0; }
};

// A constant tensor that a shape chain reads: an int32/int64 initializer, or
// the tensor carried by a Constant node.
struct ShapeExprConstant {
  std::vector<int64_t> shape;
  std::vector<int64_t> data;
};

// Reads an int32/int64 initializer into ShapeExprConstant. Returns nullopt for
// anything that is not a constant integer tensor.
std::optional<ShapeExprConstant> ReadShapeExprConstant(
    Ort::ConstValueInfo value_info);

// Builds an expression for `value_name`. `input_indices` maps fused input names
// to their index; every Shape source must already be a fused input, otherwise
// the expression cannot be recomputed inside the fusion and `expr` stays empty.
// Returns an empty expression for chains that are not provably shape-only.
// `initializer_values` carries the tensors of the initializer leaves. Names
// alone cannot be turned back into values through the C API, so a caller that
// wants initializer leaves resolved must supply them here; leaves that are
// listed in `initializer_names` but missing from `initializer_values` make the
// expression empty (the chain cannot be recomputed).
ShapeExpr BuildShapeExpr(
    const std::string& value_name,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_map<std::string, size_t>& input_indices,
    const std::unordered_set<std::string>& initializer_names,
    const std::unordered_map<std::string, ShapeExprConstant>&
        initializer_values = {});

// Evaluates the expression. `source_shapes[i]` is the runtime shape of fused
// input `source_input_indices[i]`. Returns false on any semantic error (rank,
// bounds, division by zero, element-count mismatch); callers must fall back to
// reading the tensor in that case.
bool EvaluateShapeExpr(const ShapeExpr& expr,
                       const std::vector<std::vector<int64_t>>& source_shapes,
                       std::vector<int64_t>* result);

}  // namespace musa_ep
