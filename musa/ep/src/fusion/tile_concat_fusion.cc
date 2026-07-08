// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "fusion/tile_concat_fusion.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/concat_impl.h"
#include "kernels/tensor/tile_impl.h"

namespace {

constexpr size_t kConcatManySmallInputCount = 32;
constexpr size_t kConcatManySmallMaxWidthBytes = 4096;
constexpr size_t kConcatRelaxedSmallInputCount = 2;
constexpr size_t kConcatRelaxedSmallMaxWidthBytes = 32 * 1024;
constexpr size_t kConcatManySmallMaxMapBytes = 4 * 1024 * 1024;

struct ConcatInputPlan {
  size_t data_input_index;
  size_t repeats_input_index;
  bool is_tile;
};

struct TileConcatFusionCompute : FusionNodeCompute {
  TileConcatFusionCompute(int64_t axis, size_t output_index,
                          std::vector<ConcatInputPlan> inputs)
      : axis(axis), output_index(output_index), inputs(std::move(inputs)) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      musaStream_t stream = GetComputeStream(ctx);

      std::vector<std::vector<int64_t>> shapes;
      std::vector<const void*> input_data;
      std::vector<void*> temp_buffers;
      shapes.reserve(inputs.size());
      input_data.reserve(inputs.size());

      ONNXTensorElementDataType elem_type =
          ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      size_t elem_size = 0;

      auto cleanup_temps = [&]() {
        for (void* ptr : temp_buffers) {
          FreeDeviceMemoryOnStream(ptr, stream);
        }
      };

      for (const ConcatInputPlan& input_plan : inputs) {
        Ort::ConstValue input = ctx.GetInput(input_plan.data_input_index);
        auto input_info = input.GetTensorTypeAndShapeInfo();
        auto input_shape = input_info.GetShape();
        ONNXTensorElementDataType input_elem_type = input_info.GetElementType();
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
          elem_type = input_elem_type;
          elem_size = ElementSize(elem_type);
          if (elem_size == 0) {
            return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                              "TileConcat unsupported dtype");
          }
        } else if (input_elem_type != elem_type) {
          cleanup_temps();
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "TileConcat requires uniform input dtype");
        }
        if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
          cleanup_temps();
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "TileConcat requires MUSA data inputs");
        }

        std::vector<int64_t> effective_shape = input_shape;
        const void* effective_data = input.GetTensorRawData();
        if (input_plan.is_tile) {
          std::vector<int64_t> repeats =
              ReadIntTensor(ctx, input_plan.repeats_input_index);
          if (repeats.size() < input_shape.size()) {
            repeats.insert(repeats.begin(), input_shape.size() - repeats.size(),
                           1);
          }
          if (repeats.size() > kMusaMaxBroadcastRank) {
            cleanup_temps();
            return Ort::GetApi().CreateStatus(
                ORT_NOT_IMPLEMENTED, "TileConcat Tile rank exceeds limit");
          }

          std::vector<int64_t> padded_input(repeats.size(), 1);
          const size_t offset = repeats.size() - input_shape.size();
          for (size_t i = 0; i < input_shape.size(); ++i) {
            padded_input[offset + i] = input_shape[i];
          }

          effective_shape.assign(repeats.size(), 1);
          bool identity_tile = true;
          for (size_t i = 0; i < repeats.size(); ++i) {
            if (repeats[i] < 0) {
              cleanup_temps();
              return Ort::GetApi().CreateStatus(
                  ORT_NOT_IMPLEMENTED,
                  "TileConcat requires non-negative repeats");
            }
            effective_shape[i] = padded_input[i] * repeats[i];
            identity_tile = identity_tile && repeats[i] == 1;
          }

          if (!identity_tile) {
            const size_t output_bytes =
                static_cast<size_t>(NumElements(effective_shape)) * elem_size;
            void* tile_output = nullptr;
            musaError_t alloc_status = musaMalloc(&tile_output, output_bytes);
            if (alloc_status != musaSuccess) {
              cleanup_temps();
              return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                                MusaErrorString(alloc_status));
            }
            temp_buffers.push_back(tile_output);
            OrtStatus* tile_status = LaunchStatus(LaunchMusaTileKernel(
                input.GetTensorRawData(), tile_output,
                static_cast<int32_t>(elem_size),
                MakeTileParams(input_shape, effective_shape), stream));
            if (tile_status != nullptr) {
              cleanup_temps();
              return tile_status;
            }
            effective_data = tile_output;
          }
        }

        shapes.push_back(std::move(effective_shape));
        input_data.push_back(effective_data);
      }

      if (shapes.empty()) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "TileConcat has no inputs");
      }

      int64_t normalized_axis = NormalizeAxis(axis, shapes[0].size());
      std::vector<int64_t> out_shape = shapes[0];
      out_shape[static_cast<size_t>(normalized_axis)] = 0;
      for (const auto& shape : shapes) {
        if (shape.size() != out_shape.size()) {
          cleanup_temps();
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "TileConcat input ranks must match");
        }
        for (size_t dim = 0; dim < shape.size(); ++dim) {
          if (dim == static_cast<size_t>(normalized_axis)) {
            continue;
          }
          if (shape[dim] != out_shape[dim]) {
            cleanup_temps();
            return Ort::GetApi().CreateStatus(
                ORT_NOT_IMPLEMENTED,
                "TileConcat non-axis dimensions must match");
          }
        }
        out_shape[static_cast<size_t>(normalized_axis)] +=
            shape[static_cast<size_t>(normalized_axis)];
      }

      Ort::UnownedValue output = ctx.GetOutput(output_index, out_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        cleanup_temps();
        return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                          "TileConcat requires MUSA output");
      }

      const int64_t outer =
          normalized_axis == 0
              ? 1
              : std::accumulate(out_shape.begin(),
                                out_shape.begin() + normalized_axis, int64_t{1},
                                std::multiplies<int64_t>());
      const int64_t inner =
          normalized_axis + 1 == static_cast<int64_t>(out_shape.size())
              ? 1
              : std::accumulate(out_shape.begin() + normalized_axis + 1,
                                out_shape.end(), int64_t{1},
                                std::multiplies<int64_t>());

      std::vector<int64_t> input_axis_dims(shapes.size());
      int64_t max_input_axis = 0;
      for (size_t i = 0; i < shapes.size(); ++i) {
        input_axis_dims[i] = shapes[i][static_cast<size_t>(normalized_axis)];
        max_input_axis = std::max(max_input_axis, input_axis_dims[i]);
      }

      void* output_data = output.GetTensorMutableRawData();
      const int64_t output_row_elements =
          out_shape[static_cast<size_t>(normalized_axis)] * inner;
      const size_t max_width_bytes = static_cast<size_t>(max_input_axis) *
                                     static_cast<size_t>(inner) * elem_size;
      const size_t element_descriptor_bytes =
          static_cast<size_t>(output_row_elements) *
          sizeof(MusaConcatElementDesc);

      OrtStatus* concat_status = nullptr;
      if (ShouldUseConcatSmallRows(
              input_data.size(), kConcatManySmallInputCount, max_width_bytes,
              kConcatManySmallMaxWidthBytes, output_row_elements,
              element_descriptor_bytes) ||
          ShouldUseConcatSmallRows(
              input_data.size(), kConcatRelaxedSmallInputCount, max_width_bytes,
              kConcatRelaxedSmallMaxWidthBytes, output_row_elements,
              element_descriptor_bytes)) {
        concat_status = LaunchConcatSmallRows(
            output_data, input_data, input_axis_dims, outer, inner,
            output_row_elements, static_cast<int32_t>(elem_size), stream);
      } else {
        concat_status = LaunchStatus(LaunchMusaConcatCopies(
            output_data, input_data.data(), input_axis_dims.data(),
            static_cast<int64_t>(input_data.size()), outer, inner,
            out_shape[static_cast<size_t>(normalized_axis)],
            static_cast<int32_t>(elem_size), stream));
      }

      cleanup_temps();
      return concat_status;
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

  static MusaTileParams MakeTileParams(
      const std::vector<int64_t>& input_shape,
      const std::vector<int64_t>& output_shape) {
    MusaTileParams params{};
    params.rank = static_cast<int32_t>(output_shape.size());
    params.total_elements = NumElements(output_shape);

    std::vector<int64_t> padded_input(output_shape.size(), 1);
    const size_t offset = output_shape.size() - input_shape.size();
    for (size_t i = 0; i < input_shape.size(); ++i) {
      padded_input[offset + i] = input_shape[i];
    }
    std::vector<int64_t> input_strides = Strides(input_shape);

    for (size_t dim = 0; dim < output_shape.size(); ++dim) {
      params.input_dims[dim] = padded_input[dim];
      params.output_dims[dim] = output_shape[dim];
      if (dim < offset) {
        params.input_strides[dim] = 0;
      } else {
        params.input_strides[dim] = input_strides[dim - offset];
      }
    }
    return params;
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
    musaError_t status =
        musaMalloc(reinterpret_cast<void**>(&device_element_descriptors),
                   element_descriptor_bytes);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
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
    FreeDeviceMemoryOnStream(device_element_descriptors, stream);
    return launch_status;
  }

  int64_t axis;
  size_t output_index;
  std::vector<ConcatInputPlan> inputs;
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
    throw std::runtime_error(std::string("unable to map TileConcat ") + kind +
                             " " + name);
  }
  return it->second;
}

std::unordered_map<std::string, Ort::ConstNode> BuildProducers(
    const std::vector<Ort::ConstNode>& nodes) {
  std::unordered_map<std::string, Ort::ConstNode> producers;
  for (Ort::ConstNode node : nodes) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      producers.emplace(Name(output), node);
    }
  }
  return producers;
}

}  // namespace

bool IsTileConcatFusionGraph(Ort::ConstGraph graph) {
  bool has_concat = false;
  bool has_tile = false;
  for (Ort::ConstNode node : graph.GetNodes()) {
    has_concat = has_concat || IsOnnxOp(node, "Concat");
    has_tile = has_tile || IsOnnxOp(node, "Tile");
  }
  return has_concat && has_tile;
}

std::unique_ptr<FusionNodeCompute> CreateTileConcatFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  auto producers = BuildProducers(nodes);
  auto input_indices = FusedInputIndices(fused_node);
  auto output_indices = FusedOutputIndices(fused_node);

  Ort::ConstNode concat_node{nullptr};
  for (Ort::ConstNode node : nodes) {
    if (IsOnnxOp(node, "Concat")) {
      if (concat_node) {
        throw std::runtime_error("TileConcat expects one Concat node");
      }
      concat_node = node;
    }
  }
  if (!concat_node) {
    throw std::runtime_error("TileConcat missing Concat node");
  }

  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_inputs.empty() || concat_outputs.size() != 1) {
    throw std::runtime_error("invalid TileConcat Concat node");
  }

  std::vector<ConcatInputPlan> plans;
  plans.reserve(concat_inputs.size());
  for (Ort::ConstValueInfo concat_input : concat_inputs) {
    const std::string concat_input_name = Name(concat_input);
    auto producer_it = producers.find(concat_input_name);
    if (producer_it != producers.end() &&
        IsOnnxOp(producer_it->second, "Tile")) {
      Ort::ConstNode tile_node = producer_it->second;
      std::vector<Ort::ConstValueInfo> tile_inputs = tile_node.GetInputs();
      if (tile_inputs.size() != 2) {
        throw std::runtime_error("invalid TileConcat Tile node");
      }
      plans.push_back(
          {GetIndex(input_indices, Name(tile_inputs[0]), "Tile data input"),
           GetIndex(input_indices, Name(tile_inputs[1]), "Tile repeats input"),
           true});
    } else {
      plans.push_back(
          {GetIndex(input_indices, concat_input_name, "Concat input"), 0,
           false});
    }
  }

  return std::make_unique<TileConcatFusionCompute>(
      ReadIntAttribute(concat_node, "axis", 0),
      GetIndex(output_indices, Name(concat_outputs[0]), "Concat output"),
      std::move(plans));
}
