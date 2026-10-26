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

#include "fusion/shape_expr.h"

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "graph/graph_utils.h"

// Op encodings. `attrs` is the only per-op payload beyond the input references,
// so every supported op states what it keeps there:
//
//   Constant         attrs = tensor shape, constant = row-major data
//   Shape            inputs = {source slot}; attrs = {}
//   Gather           inputs = {data, indices}; attrs = {axis}
//   Cast             inputs = {data}; attrs = {target element type}
//   Concat           inputs = {...}; attrs = {axis}
//   Unsqueeze        inputs = {data}; attrs = axes
//   Squeeze          inputs = {data}; attrs = axes (empty = drop every 1-dim)
//   Reshape          inputs = {data}; attrs = {allowzero, dims...}
//   Slice            inputs = {data}; attrs = {n, (axis, start, end) * n}
//   Add/Sub/Mul/Div/Mod
//   Floor/Ceil/Identity inputs = {..}; attrs = {}
//   ReduceProd       inputs = {data}; attrs = {keepdims, n, axes...}
//   ConstantOfShape  inputs = {shape}; attrs = {value}
//
// Only integer tensors are modelled. Every value carries its rank so that
// Gather/Slice/Concat behave the way their ONNX definitions expect.

namespace musa_ep {
namespace {

constexpr size_t kMaxShapeExprNodes = 512;
constexpr int64_t kMaxConstantElements = 4096;

struct Value {
  std::vector<int64_t> shape;
  std::vector<int64_t> data;
};

int64_t ElementCount(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      return -1;
    }
    count *= dim;
  }
  return count;
}

std::vector<int64_t> Strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t i = shape.size(); i-- > 0;) {
    if (i + 1 < shape.size()) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }
  return strides;
}

void Advance(std::vector<int64_t>* coordinate,
             const std::vector<int64_t>& shape) {
  for (size_t d = coordinate->size(); d-- > 0;) {
    if (++(*coordinate)[d] < shape[d]) {
      return;
    }
    (*coordinate)[d] = 0;
  }
}

template <typename Tensor>
bool ReadConstantTensor(const Tensor& tensor, ShapeExprConstant* out) {
  if (!tensor) {
    return false;
  }
  auto info = tensor.GetTensorTypeAndShapeInfo();
  const int64_t count = info.GetElementCount();
  if (count < 0 || count > kMaxConstantElements) {
    return false;
  }
  out->shape = info.GetShape();
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* data = tensor.template GetTensorData<int64_t>();
    out->data.assign(data, data + count);
    return true;
  }
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t* data = tensor.template GetTensorData<int32_t>();
    out->data.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
      out->data.push_back(static_cast<int64_t>(data[i]));
    }
    return true;
  }
  return false;
}

bool TryGetIntAttribute(Ort::ConstNode node, const char* name, int64_t* value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || !attr) {
    return false;
  }
  int64_t read = 0;
  status = attr.GetValue(read);
  if (!status.IsOK()) {
    return false;
  }
  *value = read;
  return true;
}

bool TryGetIntsAttribute(Ort::ConstNode node, const char* name,
                         std::vector<int64_t>* values) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || !attr) {
    return false;
  }
  std::vector<int64_t> read;
  status = attr.GetValueArray<int64_t>(read);
  if (!status.IsOK()) {
    return false;
  }
  *values = std::move(read);
  return true;
}

bool IsIntElementType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

bool NormalizeAxes(const std::vector<int64_t>& axes, int64_t rank,
                   std::vector<int64_t>* out) {
  out->clear();
  out->reserve(axes.size());
  for (int64_t axis : axes) {
    int64_t normalized = axis < 0 ? axis + rank : axis;
    if (normalized < 0 || normalized >= rank) {
      return false;
    }
    out->push_back(normalized);
  }
  std::sort(out->begin(), out->end());
  out->erase(std::unique(out->begin(), out->end()), out->end());
  return true;
}

bool BroadcastShapes(const std::vector<int64_t>& lhs,
                     const std::vector<int64_t>& rhs,
                     std::vector<int64_t>* out) {
  const size_t rank = std::max(lhs.size(), rhs.size());
  out->assign(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t lhs_dim =
        i + lhs.size() < rank ? 1 : lhs[i + lhs.size() - rank];
    const int64_t rhs_dim =
        i + rhs.size() < rank ? 1 : rhs[i + rhs.size() - rank];
    if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
      return false;
    }
    (*out)[i] = std::max(lhs_dim, rhs_dim);
  }
  return true;
}

int64_t BroadcastOffset(const std::vector<int64_t>& source_shape,
                        const std::vector<int64_t>& out_shape,
                        const std::vector<int64_t>& out_coordinate) {
  const size_t offset = out_shape.size() - source_shape.size();
  int64_t index = 0;
  for (size_t i = 0; i < source_shape.size(); ++i) {
    const int64_t coordinate =
        source_shape[i] == 1 ? 0 : out_coordinate[offset + i];
    index = index * source_shape[i] + coordinate;
  }
  return index;
}

bool EvalRef(const ShapeExpr& expr, const ShapeExprRef& ref,
             const std::vector<std::vector<int64_t>>& source_shapes,
             std::vector<std::optional<Value>>* memo, Value* out);

bool EvalNode(const ShapeExpr& expr, int index,
              const std::vector<std::vector<int64_t>>& source_shapes,
              std::vector<std::optional<Value>>* memo, Value* out) {
  if (index < 0 || static_cast<size_t>(index) >= expr.nodes.size()) {
    return false;
  }
  std::optional<Value>& slot = (*memo)[static_cast<size_t>(index)];
  if (slot.has_value()) {
    *out = *slot;
    return true;
  }

  const ShapeExprNode& node = expr.nodes[static_cast<size_t>(index)];
  Value value;

  if (node.op == "Constant") {
    value.shape = node.attrs;
    value.data = node.constant;
  } else if (node.op == "Shape") {
    if (node.inputs.size() != 1 || node.inputs[0].is_node ||
        !EvalRef(expr, node.inputs[0], source_shapes, memo, &value)) {
      return false;
    }
  } else if (node.op == "Gather") {
    if (node.inputs.size() != 2) {
      return false;
    }
    Value data;
    Value indices;
    if (!EvalRef(expr, node.inputs[0], source_shapes, memo, &data) ||
        !EvalRef(expr, node.inputs[1], source_shapes, memo, &indices)) {
      return false;
    }
    const int64_t rank = static_cast<int64_t>(data.shape.size());
    if (rank == 0) {
      return false;
    }
    int64_t axis = node.attrs.empty() ? 0 : node.attrs[0];
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      return false;
    }
    const size_t axis_index = static_cast<size_t>(axis);
    const int64_t axis_dim = data.shape[axis_index];
    int64_t inner = 1;
    for (size_t i = axis_index + 1; i < data.shape.size(); ++i) {
      inner *= data.shape[i];
    }
    value.shape.assign(data.shape.begin(), data.shape.begin() + axis);
    value.shape.insert(value.shape.end(), indices.shape.begin(),
                       indices.shape.end());
    value.shape.insert(value.shape.end(), data.shape.begin() + axis + 1,
                       data.shape.end());
    const int64_t count = ElementCount(value.shape);
    if (count < 0) {
      return false;
    }
    value.data.assign(static_cast<size_t>(count), 0);
    const std::vector<int64_t> index_shape = indices.shape;
    std::vector<int64_t> coordinate(value.shape.size(), 0);
    for (int64_t out_index = 0; out_index < count; ++out_index) {
      int64_t outer_index = 0;
      for (size_t i = 0; i < axis_index; ++i) {
        outer_index = outer_index * data.shape[i] + coordinate[i];
      }
      int64_t gathered = 0;
      for (size_t i = 0; i < index_shape.size(); ++i) {
        gathered = gathered * index_shape[i] + coordinate[axis_index + i];
      }
      int64_t position = indices.data[static_cast<size_t>(gathered)];
      if (position < 0) {
        position += axis_dim;
      }
      if (position < 0 || position >= axis_dim) {
        return false;
      }
      int64_t inner_index = 0;
      for (size_t i = axis_index + 1; i < data.shape.size(); ++i) {
        inner_index =
            inner_index * data.shape[i] +
            coordinate[axis_index + index_shape.size() + (i - axis_index - 1)];
      }
      value.data[static_cast<size_t>(out_index)] =
          data.data[static_cast<size_t>(
              (outer_index * axis_dim + position) * inner + inner_index)];
      Advance(&coordinate, value.shape);
    }
  } else if (node.op == "Reshape") {
    if (node.inputs.size() != 1 || node.attrs.empty()) {
      return false;
    }
    Value data;
    if (!EvalRef(expr, node.inputs[0], source_shapes, memo, &data)) {
      return false;
    }
    const int64_t allowzero = node.attrs[0];
    value.shape.assign(node.attrs.begin() + 1, node.attrs.end());
    for (size_t i = 0; i < value.shape.size(); ++i) {
      if (value.shape[i] == 0 && allowzero == 0) {
        if (i >= data.shape.size()) {
          return false;
        }
        value.shape[i] = data.shape[i];
      }
    }
    int64_t inferred = -1;
    for (size_t i = 0; i < value.shape.size(); ++i) {
      if (value.shape[i] == -1) {
        if (inferred >= 0) {
          return false;
        }
        inferred = static_cast<int64_t>(i);
      } else if (value.shape[i] < 0) {
        return false;
      }
    }
    const int64_t source_count = ElementCount(data.shape);
    if (source_count < 0) {
      return false;
    }
    if (inferred >= 0) {
      int64_t known = 1;
      for (int64_t dim : value.shape) {
        if (dim > 0) {
          known *= dim;
        }
      }
      if (known == 0 || source_count % known != 0) {
        return false;
      }
      value.shape[static_cast<size_t>(inferred)] = source_count / known;
    }
    if (ElementCount(value.shape) != source_count) {
      return false;
    }
    value.data = data.data;
  } else if (node.op == "Slice") {
    if (node.inputs.size() != 1 || node.attrs.empty()) {
      return false;
    }
    Value data;
    if (!EvalRef(expr, node.inputs[0], source_shapes, memo, &data)) {
      return false;
    }
    const int64_t rank = static_cast<int64_t>(data.shape.size());
    const int64_t entries = node.attrs[0];
    if (entries < 0 ||
        static_cast<size_t>(entries) * 3 + 1 != node.attrs.size()) {
      return false;
    }
    std::vector<int64_t> starts(static_cast<size_t>(rank), 0);
    std::vector<int64_t> ends = data.shape;
    for (int64_t entry = 0; entry < entries; ++entry) {
      int64_t axis = node.attrs[static_cast<size_t>(1 + entry * 3)];
      int64_t start = node.attrs[static_cast<size_t>(2 + entry * 3)];
      int64_t end = node.attrs[static_cast<size_t>(3 + entry * 3)];
      if (axis < 0) {
        axis += rank;
      }
      if (axis < 0 || axis >= rank) {
        return false;
      }
      const size_t axis_index = static_cast<size_t>(axis);
      const int64_t dim = data.shape[axis_index];
      if (start < 0) {
        start += dim;
      }
      if (end < 0) {
        end += dim;
      }
      starts[axis_index] = std::clamp<int64_t>(start, 0, dim);
      ends[axis_index] = std::clamp<int64_t>(end, 0, dim);
    }
    value.shape = data.shape;
    for (size_t i = 0; i < value.shape.size(); ++i) {
      value.shape[i] = ends[i] - starts[i];
      if (value.shape[i] < 0) {
        return false;
      }
    }
    const int64_t count = ElementCount(value.shape);
    if (count < 0) {
      return false;
    }
    const std::vector<int64_t> strides = Strides(data.shape);
    value.data.assign(static_cast<size_t>(count), 0);
    std::vector<int64_t> coordinate(value.shape.size(), 0);
    for (int64_t i = 0; i < count; ++i) {
      int64_t source_offset = 0;
      for (size_t d = 0; d < coordinate.size(); ++d) {
        source_offset += (coordinate[d] + starts[d]) * strides[d];
      }
      value.data[static_cast<size_t>(i)] =
          data.data[static_cast<size_t>(source_offset)];
      Advance(&coordinate, value.shape);
    }
  } else if (node.op == "Concat") {
    if (node.inputs.empty() || node.attrs.empty()) {
      return false;
    }
    std::vector<Value> operands(node.inputs.size());
    for (size_t i = 0; i < node.inputs.size(); ++i) {
      if (!EvalRef(expr, node.inputs[i], source_shapes, memo, &operands[i])) {
        return false;
      }
      if (operands[i].shape.size() != operands.front().shape.size()) {
        return false;
      }
    }
    const int64_t rank = static_cast<int64_t>(operands.front().shape.size());
    int64_t axis = node.attrs[0];
    if (axis < 0) {
      axis += rank;
    }
    if (rank == 0 || axis < 0 || axis >= rank) {
      return false;
    }
    const size_t axis_index = static_cast<size_t>(axis);
    value.shape = operands.front().shape;
    value.shape[axis_index] = 0;
    for (const Value& operand : operands) {
      for (size_t d = 0; d < value.shape.size(); ++d) {
        if (d == axis_index) {
          continue;
        }
        if (operand.shape[d] != value.shape[d]) {
          return false;
        }
      }
      value.shape[axis_index] += operand.shape[axis_index];
    }
    const int64_t count = ElementCount(value.shape);
    if (count < 0) {
      return false;
    }
    value.data.assign(static_cast<size_t>(count), 0);
    int64_t inner = 1;
    for (size_t i = axis_index + 1; i < value.shape.size(); ++i) {
      inner *= value.shape[i];
    }
    int64_t outer = 1;
    for (size_t i = 0; i < axis_index; ++i) {
      outer *= value.shape[i];
    }
    const int64_t row = value.shape[axis_index] * inner;
    for (int64_t o = 0; o < outer; ++o) {
      int64_t written = o * row;
      for (const Value& operand : operands) {
        const int64_t chunk = operand.shape[axis_index] * inner;
        const int64_t begin = o * chunk;
        std::copy(operand.data.begin() + begin,
                  operand.data.begin() + begin + chunk,
                  value.data.begin() + written);
        written += chunk;
      }
    }
  } else if (node.op == "Unsqueeze") {
    if (node.inputs.size() != 1) {
      return false;
    }
    Value data;
    if (!EvalRef(expr, node.inputs[0], source_shapes, memo, &data)) {
      return false;
    }
    const int64_t out_rank = static_cast<int64_t>(data.shape.size()) +
                             static_cast<int64_t>(node.attrs.size());
    std::vector<int64_t> axes;
    if (!NormalizeAxes(node.attrs, out_rank, &axes)) {
      return false;
    }
    value.shape.clear();
    size_t source = 0;
    for (int64_t i = 0; i < out_rank; ++i) {
      if (std::binary_search(axes.begin(), axes.end(), i)) {
        value.shape.push_back(1);
      } else {
        if (source >= data.shape.size()) {
          return false;
        }
        value.shape.push_back(data.shape[source++]);
      }
    }
    if (source != data.shape.size()) {
      return false;
    }
    value.data = data.data;
  } else if (node.op == "Squeeze") {
    if (node.inputs.size() != 1) {
      return false;
    }
    Value data;
    if (!EvalRef(expr, node.inputs[0], source_shapes, memo, &data)) {
      return false;
    }
    std::vector<int64_t> axes;
    if (node.attrs.empty()) {
      for (size_t i = 0; i < data.shape.size(); ++i) {
        if (data.shape[i] == 1) {
          axes.push_back(static_cast<int64_t>(i));
        }
      }
    } else if (!NormalizeAxes(node.attrs,
                              static_cast<int64_t>(data.shape.size()), &axes)) {
      return false;
    }
    value.shape.clear();
    for (size_t i = 0; i < data.shape.size(); ++i) {
      if (std::binary_search(axes.begin(), axes.end(),
                             static_cast<int64_t>(i))) {
        if (data.shape[i] != 1) {
          return false;
        }
        continue;
      }
      value.shape.push_back(data.shape[i]);
    }
    value.data = data.data;
  } else if (node.op == "ReduceProd") {
    if (node.inputs.size() != 1 || node.attrs.size() < 2) {
      return false;
    }
    Value data;
    if (!EvalRef(expr, node.inputs[0], source_shapes, memo, &data)) {
      return false;
    }
    const int64_t keepdims = node.attrs[0];
    const int64_t axis_count = node.attrs[1];
    if (axis_count < 0 ||
        static_cast<size_t>(axis_count) + 2 != node.attrs.size()) {
      return false;
    }
    std::vector<int64_t> axes;
    const int64_t rank = static_cast<int64_t>(data.shape.size());
    if (axis_count == 0) {
      for (int64_t i = 0; i < rank; ++i) {
        axes.push_back(i);
      }
    } else if (!NormalizeAxes(std::vector<int64_t>(node.attrs.begin() + 2,
                                                   node.attrs.end()),
                              rank, &axes)) {
      return false;
    }
    std::vector<bool> reduced(data.shape.size(), false);
    for (int64_t axis : axes) {
      reduced[static_cast<size_t>(axis)] = true;
    }
    value.shape.clear();
    for (size_t i = 0; i < data.shape.size(); ++i) {
      if (reduced[i]) {
        if (keepdims != 0) {
          value.shape.push_back(1);
        }
      } else {
        value.shape.push_back(data.shape[i]);
      }
    }
    const int64_t count = ElementCount(value.shape);
    const int64_t source_count = ElementCount(data.shape);
    if (count < 0 || source_count < 0) {
      return false;
    }
    value.data.assign(static_cast<size_t>(count), 1);
    std::vector<int64_t> coordinate(data.shape.size(), 0);
    for (int64_t i = 0; i < source_count; ++i) {
      int64_t out_index = 0;
      size_t out_dim = 0;
      for (size_t d = 0; d < data.shape.size(); ++d) {
        if (reduced[d]) {
          continue;
        }
        out_index = out_index * value.shape[out_dim++] + coordinate[d];
      }
      value.data[static_cast<size_t>(out_index)] *=
          data.data[static_cast<size_t>(i)];
      Advance(&coordinate, data.shape);
    }
  } else if (node.op == "ConstantOfShape") {
    if (node.inputs.size() != 1 || node.attrs.empty()) {
      return false;
    }
    Value shape_value;
    if (!EvalRef(expr, node.inputs[0], source_shapes, memo, &shape_value)) {
      return false;
    }
    value.shape = shape_value.data;
    const int64_t count = ElementCount(value.shape);
    if (count < 0) {
      return false;
    }
    value.data.assign(static_cast<size_t>(count), node.attrs[0]);
  } else if (node.op == "Cast" || node.op == "Floor" || node.op == "Ceil" ||
             node.op == "Identity") {
    if (node.inputs.size() != 1 ||
        !EvalRef(expr, node.inputs[0], source_shapes, memo, &value)) {
      return false;
    }
  } else if (node.op == "Add" || node.op == "Sub" || node.op == "Mul" ||
             node.op == "Div" || node.op == "Mod") {
    if (node.inputs.size() != 2) {
      return false;
    }
    Value lhs;
    Value rhs;
    if (!EvalRef(expr, node.inputs[0], source_shapes, memo, &lhs) ||
        !EvalRef(expr, node.inputs[1], source_shapes, memo, &rhs)) {
      return false;
    }
    if (!BroadcastShapes(lhs.shape, rhs.shape, &value.shape)) {
      return false;
    }
    const int64_t count = ElementCount(value.shape);
    if (count < 0) {
      return false;
    }
    value.data.assign(static_cast<size_t>(count), 0);
    std::vector<int64_t> coordinate(value.shape.size(), 0);
    for (int64_t i = 0; i < count; ++i) {
      const int64_t a = lhs.data[static_cast<size_t>(
          BroadcastOffset(lhs.shape, value.shape, coordinate))];
      const int64_t b = rhs.data[static_cast<size_t>(
          BroadcastOffset(rhs.shape, value.shape, coordinate))];
      int64_t result = 0;
      if (node.op == "Add") {
        result = a + b;
      } else if (node.op == "Sub") {
        result = a - b;
      } else if (node.op == "Mul") {
        result = a * b;
      } else if (node.op == "Div") {
        if (b == 0) {
          return false;
        }
        result = a / b;
      } else {
        if (b == 0) {
          return false;
        }
        result = a % b;
        if (result != 0 && (result < 0) != (b < 0)) {
          result += b;
        }
      }
      value.data[static_cast<size_t>(i)] = result;
      Advance(&coordinate, value.shape);
    }
  } else {
    return false;
  }

  const int64_t count = ElementCount(value.shape);
  if (count < 0 || static_cast<size_t>(count) != value.data.size()) {
    return false;
  }
  slot = value;
  *out = value;
  return true;
}

bool EvalRef(const ShapeExpr& expr, const ShapeExprRef& ref,
             const std::vector<std::vector<int64_t>>& source_shapes,
             std::vector<std::optional<Value>>* memo, Value* out) {
  if (!ref.is_node) {
    if (ref.index < 0 ||
        static_cast<size_t>(ref.index) >= source_shapes.size()) {
      return false;
    }
    out->shape = {static_cast<int64_t>(source_shapes[ref.index].size())};
    out->data = source_shapes[ref.index];
    return true;
  }
  return EvalNode(expr, ref.index, source_shapes, memo, out);
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

using Producers = std::unordered_map<std::string, Ort::ConstNode>;
using InputIndices = std::unordered_map<std::string, size_t>;
using InitializerValues = std::unordered_map<std::string, ShapeExprConstant>;

class ExprBuilder {
 public:
  ExprBuilder(const Producers& producers, const InputIndices& input_indices,
              const std::unordered_set<std::string>& initializer_names,
              const InitializerValues& initializer_values)
      : producers_(producers),
        input_indices_(input_indices),
        initializer_names_(initializer_names),
        initializer_values_(initializer_values) {}

  ShapeExpr Build(const std::string& value_name) {
    ShapeExpr expr;
    const int root = BuildValue(value_name, &expr);
    if (root < 0) {
      return ShapeExpr{};
    }
    expr.root = root;
    return expr;
  }

 private:
  int BuildValue(const std::string& name, ShapeExpr* expr) {
    if (name.empty()) {
      return -1;
    }
    auto memo_it = memo_.find(name);
    if (memo_it != memo_.end()) {
      return memo_it->second;
    }
    if (!pending_.insert(name).second) {
      return -1;
    }

    int result = -1;
    auto producer_it = producers_.find(name);
    if (producer_it == producers_.end()) {
      auto constant_it = initializer_values_.find(name);
      if (initializer_names_.count(name) != 0 &&
          constant_it != initializer_values_.end()) {
        ShapeExprNode node;
        node.op = "Constant";
        node.attrs = constant_it->second.shape;
        node.constant = constant_it->second.data;
        result = PushNode(std::move(node), expr);
      }
    } else {
      result = BuildNode(producer_it->second, expr);
    }

    pending_.erase(name);
    if (result < 0) {
      return -1;
    }
    memo_.emplace(name, result);
    return result;
  }

  int BuildConstantNode(Ort::ConstNode node, ShapeExpr* expr) {
    ShapeExprConstant constant;
    Ort::ConstOpAttr attr;
    if (node.GetAttributeByName("value", attr).IsOK() && attr) {
      Ort::Value tensor{nullptr};
      if (!attr.GetTensorAttributeAsOrtValue(tensor).IsOK() ||
          !ReadConstantTensor(tensor, &constant)) {
        return -1;
      }
    } else {
      int64_t int_value = 0;
      std::vector<int64_t> ints_value;
      if (TryGetIntAttribute(node, "value_int", &int_value)) {
        constant.shape = {};
        constant.data = {int_value};
      } else if (TryGetIntsAttribute(node, "value_ints", &ints_value)) {
        constant.shape = {static_cast<int64_t>(ints_value.size())};
        constant.data = std::move(ints_value);
      } else {
        return -1;
      }
    }
    ShapeExprNode out;
    out.op = "Constant";
    out.attrs = std::move(constant.shape);
    out.constant = std::move(constant.data);
    return PushNode(std::move(out), expr);
  }

  int BuildNode(Ort::ConstNode node, ShapeExpr* expr) {
    if (!node || !IsOnnxDomain(node.GetDomain())) {
      return -1;
    }
    const std::string op = node.GetOperatorType();
    const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();

    if (op == "Constant") {
      return BuildConstantNode(node, expr);
    }
    if (op == "Shape") {
      if (inputs.size() != 1) {
        return -1;
      }
      auto index_it = input_indices_.find(Name(inputs[0]));
      if (index_it == input_indices_.end()) {
        return -1;
      }
      ShapeExprNode out;
      out.op = "Shape";
      out.inputs.push_back(
          {false, static_cast<int>(SourceSlot(index_it->second, expr))});
      return PushNode(std::move(out), expr);
    }
    if (op == "Gather") {
      if (inputs.size() != 2) {
        return -1;
      }
      int64_t axis = 0;
      TryGetIntAttribute(node, "axis", &axis);
      std::vector<ShapeExprRef> refs;
      if (!AddInputRef(inputs[0], expr, &refs) ||
          !AddInputRef(inputs[1], expr, &refs)) {
        return -1;
      }
      ShapeExprNode out;
      out.op = "Gather";
      out.inputs = std::move(refs);
      out.attrs = {axis};
      return PushNode(std::move(out), expr);
    }
    if (op == "Cast") {
      if (inputs.size() != 1) {
        return -1;
      }
      int64_t to = 0;
      if (!TryGetIntAttribute(node, "to", &to) ||
          !IsIntElementType(static_cast<ONNXTensorElementDataType>(to))) {
        return -1;
      }
      std::vector<ShapeExprRef> refs;
      if (!AddInputRef(inputs[0], expr, &refs)) {
        return -1;
      }
      ShapeExprNode out;
      out.op = "Cast";
      out.inputs = std::move(refs);
      out.attrs = {to};
      return PushNode(std::move(out), expr);
    }
    if (op == "Concat") {
      if (inputs.empty()) {
        return -1;
      }
      int64_t axis = 0;
      TryGetIntAttribute(node, "axis", &axis);
      std::vector<ShapeExprRef> refs;
      for (Ort::ConstValueInfo input : inputs) {
        if (!AddInputRef(input, expr, &refs)) {
          return -1;
        }
      }
      ShapeExprNode out;
      out.op = "Concat";
      out.inputs = std::move(refs);
      out.attrs = {axis};
      return PushNode(std::move(out), expr);
    }
    if (op == "Unsqueeze" || op == "Squeeze") {
      if (inputs.empty() || inputs.size() > 2) {
        return -1;
      }
      std::vector<int64_t> axes;
      if (inputs.size() == 2) {
        if (!ReadConstantOperand(inputs[1], expr, &axes)) {
          return -1;
        }
      } else if (!TryGetIntsAttribute(node, "axes", &axes)) {
        axes.clear();  // opset < 13 Squeeze without axes drops every 1-dim
      }
      if (op == "Unsqueeze" && axes.empty()) {
        return -1;
      }
      std::vector<ShapeExprRef> refs;
      if (!AddInputRef(inputs[0], expr, &refs)) {
        return -1;
      }
      ShapeExprNode out;
      out.op = op;
      out.inputs = std::move(refs);
      out.attrs = std::move(axes);
      return PushNode(std::move(out), expr);
    }
    if (op == "Reshape") {
      if (inputs.size() < 2) {
        return -1;
      }
      std::vector<int64_t> shape;
      if (!ReadConstantOperand(inputs[1], expr, &shape)) {
        return -1;
      }
      int64_t allowzero = 0;
      TryGetIntAttribute(node, "allowzero", &allowzero);
      std::vector<ShapeExprRef> refs;
      if (!AddInputRef(inputs[0], expr, &refs)) {
        return -1;
      }
      ShapeExprNode out;
      out.op = "Reshape";
      out.inputs = std::move(refs);
      out.attrs.push_back(allowzero);
      out.attrs.insert(out.attrs.end(), shape.begin(), shape.end());
      return PushNode(std::move(out), expr);
    }
    if (op == "Slice") {
      std::vector<int64_t> starts;
      std::vector<int64_t> ends;
      std::vector<int64_t> axes;
      std::vector<int64_t> steps;
      if (TryGetIntsAttribute(node, "starts", &starts)) {
        if (!TryGetIntsAttribute(node, "ends", &ends)) {
          return -1;
        }
        TryGetIntsAttribute(node, "axes", &axes);
        TryGetIntsAttribute(node, "steps", &steps);
      } else {
        if (inputs.size() < 3) {
          return -1;
        }
        if (!ReadConstantOperand(inputs[1], expr, &starts) ||
            !ReadConstantOperand(inputs[2], expr, &ends)) {
          return -1;
        }
        if (inputs.size() > 3 && inputs[3] != nullptr &&
            !ReadConstantOperand(inputs[3], expr, &axes)) {
          return -1;
        }
        if (inputs.size() > 4 && inputs[4] != nullptr &&
            !ReadConstantOperand(inputs[4], expr, &steps)) {
          return -1;
        }
      }
      if (starts.size() != ends.size()) {
        return -1;
      }
      if (axes.empty()) {
        for (size_t i = 0; i < starts.size(); ++i) {
          axes.push_back(static_cast<int64_t>(i));
        }
      }
      if (axes.size() != starts.size()) {
        return -1;
      }
      if (!steps.empty()) {
        if (steps.size() != starts.size()) {
          return -1;
        }
        for (int64_t step : steps) {
          if (step != 1) {
            return -1;
          }
        }
      }
      std::vector<ShapeExprRef> refs;
      if (!AddInputRef(inputs[0], expr, &refs)) {
        return -1;
      }
      ShapeExprNode out;
      out.op = "Slice";
      out.inputs = std::move(refs);
      out.attrs.push_back(static_cast<int64_t>(starts.size()));
      for (size_t i = 0; i < starts.size(); ++i) {
        out.attrs.push_back(axes[i]);
        out.attrs.push_back(starts[i]);
        out.attrs.push_back(ends[i]);
      }
      return PushNode(std::move(out), expr);
    }
    if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div" ||
        op == "Mod") {
      if (inputs.size() != 2) {
        return -1;
      }
      std::vector<ShapeExprRef> refs;
      if (!AddInputRef(inputs[0], expr, &refs) ||
          !AddInputRef(inputs[1], expr, &refs)) {
        return -1;
      }
      ShapeExprNode out;
      out.op = op;
      out.inputs = std::move(refs);
      return PushNode(std::move(out), expr);
    }
    if (op == "Floor" || op == "Ceil" || op == "Identity") {
      if (inputs.size() != 1) {
        return -1;
      }
      std::vector<ShapeExprRef> refs;
      if (!AddInputRef(inputs[0], expr, &refs)) {
        return -1;
      }
      ShapeExprNode out;
      out.op = op;
      out.inputs = std::move(refs);
      return PushNode(std::move(out), expr);
    }
    if (op == "ReduceProd") {
      if (inputs.empty()) {
        return -1;
      }
      std::vector<int64_t> axes;
      if (inputs.size() > 1 && inputs[1] != nullptr) {
        if (!ReadConstantOperand(inputs[1], expr, &axes)) {
          return -1;
        }
      } else {
        TryGetIntsAttribute(node, "axes", &axes);
      }
      int64_t keepdims = 1;
      TryGetIntAttribute(node, "keepdims", &keepdims);
      std::vector<ShapeExprRef> refs;
      if (!AddInputRef(inputs[0], expr, &refs)) {
        return -1;
      }
      ShapeExprNode out;
      out.op = "ReduceProd";
      out.inputs = std::move(refs);
      out.attrs.push_back(keepdims);
      out.attrs.push_back(static_cast<int64_t>(axes.size()));
      out.attrs.insert(out.attrs.end(), axes.begin(), axes.end());
      return PushNode(std::move(out), expr);
    }
    if (op == "ConstantOfShape") {
      if (inputs.size() != 1) {
        return -1;
      }
      std::vector<ShapeExprRef> refs;
      if (!AddInputRef(inputs[0], expr, &refs)) {
        return -1;
      }
      int64_t fill = 0;
      Ort::ConstOpAttr attr;
      if (node.GetAttributeByName("value", attr).IsOK() && attr) {
        Ort::Value tensor{nullptr};
        ShapeExprConstant constant;
        if (!attr.GetTensorAttributeAsOrtValue(tensor).IsOK() ||
            !ReadConstantTensor(tensor, &constant) ||
            constant.data.size() != 1) {
          return -1;
        }
        fill = constant.data[0];
      }
      ShapeExprNode out;
      out.op = op;
      out.inputs = std::move(refs);
      out.attrs = {fill};
      return PushNode(std::move(out), expr);
    }
    return -1;
  }

  bool AddInputRef(Ort::ConstValueInfo input, ShapeExpr* expr,
                   std::vector<ShapeExprRef>* refs) {
    if (input == nullptr) {
      return false;
    }
    const int index = BuildValue(Name(input), expr);
    if (index < 0) {
      return false;
    }
    refs->push_back({true, index});
    return true;
  }

  bool ReadConstantOperand(Ort::ConstValueInfo input, ShapeExpr* expr,
                           std::vector<int64_t>* data) {
    if (input == nullptr) {
      return false;
    }
    const int index = BuildValue(Name(input), expr);
    if (index < 0) {
      return false;
    }
    const std::vector<std::vector<int64_t>> no_sources;
    std::vector<std::optional<Value>> memo(expr->nodes.size());
    Value value;
    if (!EvalNode(*expr, index, no_sources, &memo, &value)) {
      return false;
    }
    *data = std::move(value.data);
    return true;
  }

  size_t SourceSlot(size_t fused_input_index, ShapeExpr* expr) {
    for (size_t slot = 0; slot < expr->source_input_indices.size(); ++slot) {
      if (expr->source_input_indices[slot] == fused_input_index) {
        return slot;
      }
    }
    expr->source_input_indices.push_back(fused_input_index);
    return expr->source_input_indices.size() - 1;
  }

  int PushNode(ShapeExprNode node, ShapeExpr* expr) {
    if (expr->nodes.size() >= kMaxShapeExprNodes) {
      return -1;
    }
    expr->nodes.push_back(std::move(node));
    return static_cast<int>(expr->nodes.size() - 1);
  }

  const Producers& producers_;
  const InputIndices& input_indices_;
  const std::unordered_set<std::string>& initializer_names_;
  const InitializerValues& initializer_values_;
  std::unordered_map<std::string, int> memo_;
  std::unordered_set<std::string> pending_;
};

}  // namespace

std::optional<ShapeExprConstant> ReadShapeExprConstant(
    Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    return std::nullopt;
  }
  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    return std::nullopt;
  }
  ShapeExprConstant constant;
  if (!ReadConstantTensor(value, &constant)) {
    return std::nullopt;
  }
  return constant;
}

ShapeExpr BuildShapeExpr(
    const std::string& value_name,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_map<std::string, size_t>& input_indices,
    const std::unordered_set<std::string>& initializer_names,
    const std::unordered_map<std::string, ShapeExprConstant>&
        initializer_values) {
  ExprBuilder builder(producers, input_indices, initializer_names,
                      initializer_values);
  return builder.Build(value_name);
}

bool EvaluateShapeExpr(const ShapeExpr& expr,
                       const std::vector<std::vector<int64_t>>& source_shapes,
                       std::vector<int64_t>* result) {
  if (expr.Empty() || result == nullptr ||
      expr.source_input_indices.size() != source_shapes.size()) {
    return false;
  }
  std::vector<std::optional<Value>> memo(expr.nodes.size());
  Value value;
  if (!EvalNode(expr, expr.root, source_shapes, &memo, &value)) {
    return false;
  }
  *result = std::move(value.data);
  return true;
}

}  // namespace musa_ep
