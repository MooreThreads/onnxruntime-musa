// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "fusion/concat_reshape_fusion.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/concat_impl.h"

namespace {

constexpr size_t kConcatManySmallInputCount = 32;
constexpr size_t kConcatManySmallMaxWidthBytes = 4096;
constexpr size_t kConcatRelaxedSmallInputCount = 2;
constexpr size_t kConcatRelaxedSmallMaxWidthBytes = 32 * 1024;
constexpr size_t kConcatManySmallMaxMapBytes = 4 * 1024 * 1024;

struct ConcatReshapeFusionCompute : FusionNodeCompute {
  ConcatReshapeFusionCompute(int64_t axis, int64_t allowzero,
                             std::vector<int64_t> unsqueeze_axes,
                             std::vector<int64_t> target_shape,
                             size_t output_index,
                             std::vector<size_t> input_indices)
      : axis(axis),
        allowzero(allowzero),
        unsqueeze_axes(std::move(unsqueeze_axes)),
        target_shape(std::move(target_shape)),
        output_index(output_index),
        input_indices(std::move(input_indices)) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      std::vector<std::vector<int64_t>> shapes;
      std::vector<const void*> input_data;
      std::vector<std::unique_ptr<DeviceInputBuffer>> input_buffers;
      shapes.reserve(input_indices.size());
      input_data.reserve(input_indices.size());
      input_buffers.reserve(input_indices.size());
      musaStream_t stream = GetComputeStream(ctx);

      ONNXTensorElementDataType elem_type =
          ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      size_t elem_size = 0;
      for (size_t input_index : input_indices) {
        Ort::ConstValue input = ctx.GetInput(input_index);
        auto input_info = input.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType input_elem_type = input_info.GetElementType();
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
          elem_type = input_elem_type;
          elem_size = ElementSize(elem_type);
          if (elem_size == 0) {
            return Ort::GetApi().CreateStatus(
                ORT_NOT_IMPLEMENTED, "ConcatReshape unsupported dtype");
          }
        } else if (input_elem_type != elem_type) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED,
              "ConcatReshape requires uniform input dtype");
        }
        shapes.push_back(input_info.GetShape());
        auto input_buffer = std::make_unique<DeviceInputBuffer>();
        RETURN_IF_ERROR(input_buffer->Bind(input, stream));
        input_data.push_back(input_buffer->data());
        input_buffers.push_back(std::move(input_buffer));
      }
      if (shapes.empty()) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "ConcatReshape has no inputs");
      }

      int64_t normalized_axis = NormalizeAxis(axis, shapes[0].size());
      std::vector<int64_t> concat_shape = shapes[0];
      concat_shape[static_cast<size_t>(normalized_axis)] = 0;
      for (const auto& shape : shapes) {
        if (shape.size() != concat_shape.size()) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "ConcatReshape input ranks must match");
        }
        for (size_t dim = 0; dim < shape.size(); ++dim) {
          if (dim == static_cast<size_t>(normalized_axis)) {
            continue;
          }
          if (shape[dim] != concat_shape[dim]) {
            return Ort::GetApi().CreateStatus(
                ORT_NOT_IMPLEMENTED,
                "ConcatReshape non-axis dimensions must match");
          }
        }
        concat_shape[static_cast<size_t>(normalized_axis)] +=
            shape[static_cast<size_t>(normalized_axis)];
      }

      std::vector<int64_t> reshape_input_shape =
          ApplyUnsqueeze(concat_shape, unsqueeze_axes);
      std::vector<int64_t> output_shape = ResolveReshapeOutputShape(
          reshape_input_shape, target_shape, allowzero);
      if (NumElements(output_shape) != NumElements(concat_shape)) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "ConcatReshape output element count mismatch");
      }

      Ort::UnownedValue output = ctx.GetOutput(output_index, output_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                          "ConcatReshape requires MUSA output");
      }

      const int64_t outer =
          normalized_axis == 0
              ? 1
              : std::accumulate(concat_shape.begin(),
                                concat_shape.begin() + normalized_axis,
                                int64_t{1}, std::multiplies<int64_t>());
      const int64_t inner =
          normalized_axis + 1 == static_cast<int64_t>(concat_shape.size())
              ? 1
              : std::accumulate(concat_shape.begin() + normalized_axis + 1,
                                concat_shape.end(), int64_t{1},
                                std::multiplies<int64_t>());

      std::vector<int64_t> input_axis_dims(shapes.size());
      int64_t max_input_axis = 0;
      for (size_t i = 0; i < shapes.size(); ++i) {
        input_axis_dims[i] = shapes[i][static_cast<size_t>(normalized_axis)];
        max_input_axis = std::max(max_input_axis, input_axis_dims[i]);
      }

      void* output_data = output.GetTensorMutableRawData();
      const int64_t output_row_elements =
          concat_shape[static_cast<size_t>(normalized_axis)] * inner;
      const size_t max_width_bytes = static_cast<size_t>(max_input_axis) *
                                     static_cast<size_t>(inner) * elem_size;
      const size_t element_descriptor_bytes =
          static_cast<size_t>(output_row_elements) *
          sizeof(MusaConcatElementDesc);
      if (ShouldUseConcatSmallRows(
              input_data.size(), kConcatManySmallInputCount, max_width_bytes,
              kConcatManySmallMaxWidthBytes, output_row_elements,
              element_descriptor_bytes) ||
          ShouldUseConcatSmallRows(
              input_data.size(), kConcatRelaxedSmallInputCount, max_width_bytes,
              kConcatRelaxedSmallMaxWidthBytes, output_row_elements,
              element_descriptor_bytes)) {
        return LaunchConcatSmallRows(output_data, input_data, input_axis_dims,
                                     outer, inner, output_row_elements,
                                     static_cast<int32_t>(elem_size), stream);
      }

      return LaunchStatus(LaunchMusaConcatCopies(
          output_data, input_data.data(), input_axis_dims.data(),
          static_cast<int64_t>(input_data.size()), outer, inner,
          concat_shape[static_cast<size_t>(normalized_axis)],
          static_cast<int32_t>(elem_size), stream));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  static bool ShouldUseConcatSmallRows(size_t input_count,
                                       size_t min_input_count,
                                       size_t max_width_bytes,
                                       size_t width_limit_bytes,
                                       int64_t output_row_elements,
                                       size_t element_descriptor_bytes) {
    return input_count >= min_input_count &&
           max_width_bytes <= width_limit_bytes && output_row_elements > 0 &&
           element_descriptor_bytes <= kConcatManySmallMaxMapBytes;
  }

  static OrtStatus* LaunchConcatSmallRows(
      void* output, const std::vector<const void*>& input_data,
      const std::vector<int64_t>& input_axis_dims, int64_t outer, int64_t inner,
      int64_t output_row_elements, int32_t elem_size, musaStream_t stream) {
    if (input_data.size() <=
        static_cast<size_t>(kMusaConcatSmallRowsMaxInputs)) {
      return LaunchStatus(LaunchMusaConcatManySmallRowsDirect(
          output, input_data.data(), input_axis_dims.data(),
          static_cast<int64_t>(input_data.size()), outer, inner,
          output_row_elements, elem_size, stream));
    }

    std::vector<MusaConcatElementDesc> element_descriptors(
        static_cast<size_t>(output_row_elements));
    int64_t output_offset = 0;
    for (size_t input_idx = 0; input_idx < input_data.size(); ++input_idx) {
      const int64_t input_width = input_axis_dims[input_idx] * inner;
      for (int64_t local_element = 0; local_element < input_width;
           ++local_element) {
        element_descriptors[static_cast<size_t>(
            output_offset + local_element)] = MusaConcatElementDesc{
            input_data[input_idx], input_width, local_element};
      }
      output_offset += input_width;
    }

    MusaConcatElementDesc* device_element_descriptors = nullptr;
    const size_t element_descriptor_bytes =
        static_cast<size_t>(output_row_elements) *
        sizeof(MusaConcatElementDesc);
    device_element_descriptors = reinterpret_cast<MusaConcatElementDesc*>(
        AllocateDeviceMemoryOnStream(element_descriptor_bytes, stream));
    if (device_element_descriptors == nullptr) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL, MusaErrorString(musaErrorMemoryAllocation));
    }

    OrtStatus* copy_status = CopyTemporaryHostToDevice(
        device_element_descriptors, element_descriptors.data(),
        element_descriptor_bytes, stream);
    if (copy_status != nullptr) {
      (void)musaFree(device_element_descriptors);
      return copy_status;
    }

    OrtStatus* launch_status = LaunchStatus(
        LaunchMusaConcatManySmallRows(output, device_element_descriptors, outer,
                                      output_row_elements, elem_size, stream));
    FreeDeviceMemoryOnStream(device_element_descriptors, stream,
                             element_descriptor_bytes);
    return launch_status;
  }

  static std::vector<int64_t> ResolveReshapeOutputShape(
      const std::vector<int64_t>& input_shape, std::vector<int64_t> out_shape,
      int64_t allowzero) {
    int64_t input_size = NumElements(input_shape);
    int64_t known = 1;
    int infer_idx = -1;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      if (out_shape[i] == 0 && !allowzero) {
        if (i >= input_shape.size()) {
          throw std::runtime_error("ConcatReshape zero dim exceeds input rank");
        }
        out_shape[i] = input_shape[i];
      }
      if (out_shape[i] == -1) {
        if (infer_idx >= 0) {
          throw std::runtime_error(
              "ConcatReshape target has multiple inferred dims");
        }
        infer_idx = static_cast<int>(i);
      } else {
        known *= out_shape[i];
      }
    }
    if (infer_idx >= 0) {
      if (known == 0 || input_size % known != 0) {
        throw std::runtime_error("ConcatReshape cannot infer output dim");
      }
      out_shape[static_cast<size_t>(infer_idx)] = input_size / known;
    }
    return out_shape;
  }

  static std::vector<int64_t> ApplyUnsqueeze(std::vector<int64_t> shape,
                                             std::vector<int64_t> axes) {
    if (axes.empty()) {
      return shape;
    }
    const int64_t output_rank =
        static_cast<int64_t>(shape.size() + axes.size());
    for (int64_t& axis : axes) {
      if (axis < 0) {
        axis += output_rank;
      }
      if (axis < 0 || axis >= output_rank) {
        throw std::runtime_error("ConcatReshape Unsqueeze axis out of range");
      }
    }
    std::sort(axes.begin(), axes.end());
    if (std::adjacent_find(axes.begin(), axes.end()) != axes.end()) {
      throw std::runtime_error("ConcatReshape Unsqueeze axes must be unique");
    }
    for (int64_t axis : axes) {
      shape.insert(shape.begin() + axis, 1);
    }
    return shape;
  }

  int64_t axis;
  int64_t allowzero;
  std::vector<int64_t> unsqueeze_axes;
  std::vector<int64_t> target_shape;
  size_t output_index;
  std::vector<size_t> input_indices;
};

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(Ort::ConstNode node, const char* op_type) {
  return node && node.GetOperatorType() == op_type &&
         IsOnnxDomain(node.GetDomain());
}

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info ? value_info.GetName() : std::string{};
}

bool HasOnlyConsumer(Ort::ConstValueInfo output, Ort::ConstNode expected_node,
                     int64_t expected_input_index) {
  if (output == nullptr) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
      output.GetConsumers();
  return consumers.size() == 1 &&
         consumers[0].node.GetId() == expected_node.GetId() &&
         consumers[0].index == expected_input_index;
}

bool IsSmallIntegerInitializer(Ort::ConstValueInfo input) {
  if (input == nullptr || !input.IsConstantInitializer()) {
    return false;
  }

  Ort::ConstTypeInfo type_info = input.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  ONNXTensorElementDataType elem_type =
      type_info.GetTensorTypeAndShapeInfo().GetElementType();
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

std::unordered_map<std::string, Ort::ConstNode> BuildProducerMap(
    const std::vector<Ort::ConstNode>& nodes) {
  std::unordered_map<std::string, Ort::ConstNode> producers;
  for (Ort::ConstNode node : nodes) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      producers.emplace(Name(output), node);
    }
  }
  return producers;
}

Ort::ConstNode FindProducer(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo input) {
  auto it = producers.find(Name(input));
  return it == producers.end() ? Ort::ConstNode{nullptr} : it->second;
}

bool IsStrictConcatReshapeGraph(Ort::ConstGraph graph) {
  std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  auto producers = BuildProducerMap(nodes);

  Ort::ConstNode concat_node{nullptr};
  Ort::ConstNode unsqueeze_node{nullptr};
  Ort::ConstNode reshape_node{nullptr};
  std::unordered_set<size_t> pattern_node_ids;

  for (Ort::ConstNode node : nodes) {
    if (IsOnnxOp(node, "Reshape")) {
      if (reshape_node) {
        return false;
      }
      reshape_node = node;
    }
  }
  if (!reshape_node) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reshape_outputs = reshape_node.GetOutputs();
  if (reshape_inputs.size() != 2 || reshape_outputs.size() != 1 ||
      !IsSmallIntegerInitializer(reshape_inputs[1])) {
    return false;
  }

  Ort::ConstValueInfo concat_or_unsqueeze_output = reshape_inputs[0];
  concat_node = FindProducer(producers, reshape_inputs[0]);
  if (IsOnnxOp(concat_node, "Unsqueeze")) {
    unsqueeze_node = concat_node;
    std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
        unsqueeze_node.GetInputs();
    std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
        unsqueeze_node.GetOutputs();
    if (unsqueeze_inputs.size() != 2 || unsqueeze_outputs.size() != 1 ||
        !IsSmallIntegerInitializer(unsqueeze_inputs[1]) ||
        !HasOnlyConsumer(unsqueeze_outputs[0], reshape_node, 0)) {
      return false;
    }
    concat_node = FindProducer(producers, unsqueeze_inputs[0]);
    concat_or_unsqueeze_output = unsqueeze_inputs[0];
    pattern_node_ids.insert(unsqueeze_node.GetId());
  }

  if (!IsOnnxOp(concat_node, "Concat")) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_inputs.empty() || concat_outputs.size() != 1 ||
      !HasOnlyConsumer(concat_outputs[0],
                       unsqueeze_node ? unsqueeze_node : reshape_node, 0) ||
      Name(concat_outputs[0]) != Name(concat_or_unsqueeze_output)) {
    return false;
  }

  for (Ort::ConstValueInfo concat_input : concat_inputs) {
    if (concat_input == nullptr || concat_input.IsConstantInitializer() ||
        IsOnnxOp(FindProducer(producers, concat_input), "Constant")) {
      return false;
    }
  }

  pattern_node_ids.insert(concat_node.GetId());
  pattern_node_ids.insert(reshape_node.GetId());
  for (Ort::ConstNode node : nodes) {
    if (pattern_node_ids.count(node.GetId()) != 0 ||
        IsOnnxOp(node, "Constant")) {
      continue;
    }
    return false;
  }
  return true;
}

int64_t ReadIntAttribute(Ort::ConstNode node, const std::string& name,
                         int64_t default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK()) {
    return default_value;
  }

  int64_t value = default_value;
  status = attr.GetValue(value);
  return status.IsOK() ? value : default_value;
}

std::vector<int64_t> ReadIntInitializer(Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    throw std::runtime_error("ConcatReshape requires constant Reshape shape");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("ConcatReshape failed to read Reshape shape");
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  const size_t count = static_cast<size_t>(info.GetElementCount());
  if (count > 100) {
    throw std::runtime_error("ConcatReshape Reshape shape is too large");
  }
  std::vector<int64_t> result;
  result.reserve(count);
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* data = value.GetTensorData<int64_t>();
    result.assign(data, data + count);
  } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t* data = value.GetTensorData<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      result.push_back(static_cast<int64_t>(data[i]));
    }
  } else {
    throw std::runtime_error("ConcatReshape Reshape shape must be int32/int64");
  }
  return result;
}

std::unordered_map<std::string, size_t> FusedInputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> indices;
  std::vector<Ort::ConstValueInfo> inputs = fused_node.GetInputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    indices.emplace(Name(inputs[i]), i);
  }
  return indices;
}

std::unordered_map<std::string, size_t> FusedOutputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> indices;
  std::vector<Ort::ConstValueInfo> outputs = fused_node.GetOutputs();
  for (size_t i = 0; i < outputs.size(); ++i) {
    indices.emplace(Name(outputs[i]), i);
  }
  return indices;
}

size_t GetIndex(const std::unordered_map<std::string, size_t>& indices,
                const std::string& name, const char* kind) {
  auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error(std::string("unable to map ConcatReshape ") +
                             kind + " " + name);
  }
  return it->second;
}

}  // namespace

bool IsConcatReshapeFusionGraph(Ort::ConstGraph graph) {
  return IsStrictConcatReshapeGraph(graph);
}

std::unique_ptr<FusionNodeCompute> CreateConcatReshapeFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  auto input_indices = FusedInputIndices(fused_node);
  auto output_indices = FusedOutputIndices(fused_node);

  Ort::ConstNode concat_node{nullptr};
  Ort::ConstNode unsqueeze_node{nullptr};
  Ort::ConstNode reshape_node{nullptr};
  for (Ort::ConstNode node : nodes) {
    if (IsOnnxOp(node, "Concat")) {
      if (concat_node) {
        throw std::runtime_error("ConcatReshape expects one Concat node");
      }
      concat_node = node;
    } else if (IsOnnxOp(node, "Unsqueeze")) {
      if (unsqueeze_node) {
        throw std::runtime_error("ConcatReshape expects at most one Unsqueeze");
      }
      unsqueeze_node = node;
    } else if (IsOnnxOp(node, "Reshape")) {
      if (reshape_node) {
        throw std::runtime_error("ConcatReshape expects one Reshape node");
      }
      reshape_node = node;
    }
  }
  if (!concat_node || !reshape_node) {
    throw std::runtime_error("ConcatReshape missing Concat or Reshape node");
  }

  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reshape_outputs = reshape_node.GetOutputs();
  if (concat_inputs.empty() || reshape_inputs.size() != 2 ||
      reshape_outputs.size() != 1) {
    throw std::runtime_error("invalid ConcatReshape graph");
  }

  std::vector<size_t> data_input_indices;
  data_input_indices.reserve(concat_inputs.size());
  for (Ort::ConstValueInfo concat_input : concat_inputs) {
    data_input_indices.push_back(
        GetIndex(input_indices, Name(concat_input), "Concat input"));
  }

  return std::make_unique<ConcatReshapeFusionCompute>(
      ReadIntAttribute(concat_node, "axis", 0),
      ReadIntAttribute(reshape_node, "allowzero", 0),
      unsqueeze_node ? ReadIntInitializer(unsqueeze_node.GetInputs()[1])
                     : std::vector<int64_t>{},
      ReadIntInitializer(reshape_inputs[1]),
      GetIndex(output_indices, Name(reshape_outputs[0]), "Reshape output"),
      std::move(data_input_indices));
}
