// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/shape_cast_transpose_fusion.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "kernels/shared_inc/blas_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/transpose_impl.h"
#include "plugin_ep_utils.h"

namespace {

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(Ort::ConstNode node, const char* op_type) {
  return node.GetOperatorType() == op_type && IsOnnxDomain(node.GetDomain());
}

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
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

std::vector<int64_t> ReadIntsAttribute(Ort::ConstNode node,
                                       const std::string& name) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK()) {
    return {};
  }

  std::vector<int64_t> values;
  status = attr.GetValueArray(values);
  return status.IsOK() ? values : std::vector<int64_t>{};
}

std::vector<int64_t> ReadIntInitializer(Ort::ConstValueInfo value_info) {
  if (!value_info || !value_info.IsConstantInitializer()) {
    throw std::runtime_error("ShapeCastTranspose requires int initializer");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("ShapeCastTranspose failed to read initializer");
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return ReadTyped<int64_t>(value);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> vals = ReadTyped<int32_t>(value);
    return std::vector<int64_t>(vals.begin(), vals.end());
  }
  throw std::runtime_error("ShapeCastTranspose constants must be int32/int64");
}

std::unordered_map<std::string, size_t> FusedInputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> fused_input_indices;
  std::vector<Ort::ConstValueInfo> fused_inputs = fused_node.GetInputs();
  for (size_t i = 0; i < fused_inputs.size(); ++i) {
    fused_input_indices.emplace(Name(fused_inputs[i]), i);
  }
  return fused_input_indices;
}

size_t GetFusedInputIndex(
    const std::unordered_map<std::string, size_t>& fused_input_indices,
    const std::string& input_name) {
  auto it = fused_input_indices.find(input_name);
  if (it == fused_input_indices.end()) {
    throw std::runtime_error("unable to map ShapeCastTranspose input " +
                             input_name);
  }
  return it->second;
}

std::unordered_map<std::string, size_t> FusedOutputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> fused_output_indices;
  std::vector<Ort::ConstValueInfo> fused_outputs = fused_node.GetOutputs();
  for (size_t i = 0; i < fused_outputs.size(); ++i) {
    fused_output_indices.emplace(Name(fused_outputs[i]), i);
  }
  return fused_output_indices;
}

size_t GetFusedOutputIndex(
    const std::unordered_map<std::string, size_t>& fused_output_indices,
    const std::string& output_name) {
  auto it = fused_output_indices.find(output_name);
  if (it == fused_output_indices.end()) {
    throw std::runtime_error("unable to map ShapeCastTranspose output " +
                             output_name);
  }
  return it->second;
}

Ort::ConstNode ProducerOf(Ort::ConstValueInfo value_info) {
  Ort::ValueInfoConsumerProducerInfo producer = value_info.GetProducerNode();
  return producer.node;
}

std::vector<int64_t> ResolveReshapeOutputShape(
    const std::vector<int64_t>& input_shape, std::vector<int64_t> out_shape,
    int64_t allowzero) {
  int64_t input_size = NumElements(input_shape);
  int64_t known = 1;
  int64_t infer_idx = -1;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    if (out_shape[i] == 0 && !allowzero) {
      if (i >= input_shape.size()) {
        throw std::runtime_error(
            "ShapeCastTranspose zero dim exceeds input rank");
      }
      out_shape[i] = input_shape[i];
    }
    if (out_shape[i] == -1) {
      if (infer_idx >= 0) {
        throw std::runtime_error(
            "ShapeCastTranspose only supports one inferred dim");
      }
      infer_idx = static_cast<int64_t>(i);
    } else {
      known *= out_shape[i];
    }
  }
  if (infer_idx >= 0) {
    if (known == 0 || input_size % known != 0) {
      throw std::runtime_error("ShapeCastTranspose cannot infer output dim");
    }
    out_shape[static_cast<size_t>(infer_idx)] = input_size / known;
  }
  if (NumElements(out_shape) != input_size) {
    throw std::runtime_error("ShapeCastTranspose element count mismatch");
  }
  return out_shape;
}

std::vector<ShapeCastTransposeTerm> BuildShapeTerms(
    Ort::ConstNode concat_node) {
  std::vector<ShapeCastTransposeTerm> terms;
  int64_t dynamic_term_count = 0;
  int64_t infer_count = 0;
  for (Ort::ConstValueInfo input : concat_node.GetInputs()) {
    if (!input.IsConstantInitializer()) {
      ++dynamic_term_count;
      terms.push_back(ShapeCastTransposeTerm{true, {}});
      continue;
    }

    ShapeCastTransposeTerm term;
    term.values = ReadIntInitializer(input);
    infer_count += static_cast<int64_t>(
        std::count(term.values.begin(), term.values.end(), -1));
    terms.push_back(std::move(term));
  }
  if (terms.empty() || dynamic_term_count > 1 || infer_count > 1 ||
      dynamic_term_count == 0) {
    throw std::runtime_error(
        "ShapeCastTranspose requires one dynamic batch shape term");
  }
  return terms;
}

std::vector<int64_t> ResolveShapeTerms(
    const std::vector<ShapeCastTransposeTerm>& terms,
    const std::vector<int64_t>& data_shape) {
  if (data_shape.empty()) {
    throw std::runtime_error("ShapeCastTranspose requires ranked data input");
  }
  std::vector<int64_t> values;
  for (const ShapeCastTransposeTerm& term : terms) {
    if (term.from_data_dim0) {
      values.push_back(data_shape[0]);
    } else {
      values.insert(values.end(), term.values.begin(), term.values.end());
    }
  }
  return values;
}

std::vector<int64_t> OutputShapeForPerm(const std::vector<int64_t>& input_shape,
                                        const std::vector<int64_t>& perm) {
  if (input_shape.size() != perm.size()) {
    throw std::runtime_error("ShapeCastTranspose perm rank mismatch");
  }
  std::vector<int64_t> out_shape;
  out_shape.reserve(perm.size());
  for (int64_t axis : perm) {
    if (axis < 0 || axis >= static_cast<int64_t>(input_shape.size())) {
      throw std::runtime_error("ShapeCastTranspose perm axis out of range");
    }
    out_shape.push_back(input_shape[static_cast<size_t>(axis)]);
  }
  return out_shape;
}

MusaTransposeParams MakeTransposeParams(const std::vector<int64_t>& input_shape,
                                        const std::vector<int64_t>& output_shape,
                                        const std::vector<int64_t>& perm) {
  if (input_shape.size() > kMusaMaxBroadcastRank ||
      input_shape.size() != output_shape.size() ||
      input_shape.size() != perm.size()) {
    throw std::runtime_error("ShapeCastTranspose unsupported rank");
  }

  auto input_strides = Strides(input_shape);
  MusaTransposeParams params{};
  params.rank = static_cast<int32_t>(input_shape.size());
  params.total_elements = NumElements(output_shape);
  for (size_t dim = 0; dim < input_shape.size(); ++dim) {
    params.input_strides[dim] = input_strides[dim];
    params.output_dims[dim] = output_shape[dim];
    params.perm[dim] = static_cast<int32_t>(perm[dim]);
  }
  return params;
}

bool TryMudnnTranspose(Ort::ConstValue input, Ort::UnownedValue output,
                       const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& output_shape,
                       const std::vector<int64_t>& perm,
                       ONNXTensorElementDataType elem_type) {
  if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo()) || input_shape.size() > 8 ||
      input_shape.size() != perm.size()) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle);
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor input_tensor;
  ::musa::dnn::Tensor output_tensor;
  if (!SetMudnnTensor(input_tensor, input.GetTensorRawData(), input_shape,
                      elem_type) ||
      !SetMudnnTensor(output_tensor, output.GetTensorMutableRawData(),
                      output_shape, elem_type)) {
    return false;
  }

  ::musa::dnn::Permute op;
  if (op.ConfigDimStride(output_tensor, input_tensor,
                         static_cast<int>(perm.size()), perm.data()) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, input_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

Ort::ConstNode FindConcatNode(Ort::ConstGraph graph) {
  Ort::ConstNode concat_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, "Concat")) {
      continue;
    }
    if (concat_node) {
      throw std::runtime_error("ShapeCastTranspose expects one Concat node");
    }
    concat_node = node;
  }
  if (!concat_node) {
    throw std::runtime_error("ShapeCastTranspose expects a Concat node");
  }
  return concat_node;
}

std::vector<ShapeCastTransposePlan> BuildOutputPlans(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode concat_node = FindConcatNode(graph);
  auto fused_input_indices = FusedInputIndices(fused_node);
  auto fused_output_indices = FusedOutputIndices(fused_node);
  std::vector<ShapeCastTransposeTerm> shape_terms =
      BuildShapeTerms(concat_node);

  std::vector<ShapeCastTransposePlan> outputs;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, "Transpose")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> transpose_inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> transpose_outputs = node.GetOutputs();
    if (transpose_inputs.size() != 1 || transpose_outputs.size() != 1) {
      throw std::runtime_error("ShapeCastTranspose invalid Transpose node");
    }
    Ort::ConstNode reshape_node = ProducerOf(transpose_inputs[0]);
    if (!reshape_node || !IsOnnxOp(reshape_node, "Reshape")) {
      throw std::runtime_error(
          "ShapeCastTranspose expects Transpose input from Reshape");
    }
    std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
    if (reshape_inputs.size() != 2) {
      throw std::runtime_error("ShapeCastTranspose invalid Reshape node");
    }

    ShapeCastTransposePlan plan;
    plan.data_input_index =
        GetFusedInputIndex(fused_input_indices, Name(reshape_inputs[0]));
    plan.output_index =
        GetFusedOutputIndex(fused_output_indices, Name(transpose_outputs[0]));
    plan.shape_terms = shape_terms;
    plan.perm = ReadIntsAttribute(node, "perm");
    plan.allowzero = ReadIntAttribute(reshape_node, "allowzero", 0);
    outputs.push_back(std::move(plan));
  }
  if (outputs.empty()) {
    throw std::runtime_error("ShapeCastTranspose requires Transpose outputs");
  }
  return outputs;
}

}  // namespace

ShapeCastTransposeFusionCompute::ShapeCastTransposeFusionCompute(
    std::vector<ShapeCastTransposePlan> outputs)
    : outputs(std::move(outputs)) {}

OrtStatus* ShapeCastTransposeFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    for (const ShapeCastTransposePlan& plan : outputs) {
      Ort::ConstValue data = ctx.GetInput(plan.data_input_index);
      auto data_info = data.GetTensorTypeAndShapeInfo();
      auto elem_type = data_info.GetElementType();
      std::vector<int64_t> data_shape = data_info.GetShape();
      std::vector<int64_t> reshape_shape = ResolveReshapeOutputShape(
          data_shape, ResolveShapeTerms(plan.shape_terms, data_shape),
          plan.allowzero);

      std::vector<int64_t> perm = plan.perm;
      if (perm.empty()) {
        for (int64_t i = static_cast<int64_t>(reshape_shape.size()) - 1;
             i >= 0; --i) {
          perm.push_back(i);
        }
      }
      std::vector<int64_t> output_shape =
          OutputShapeForPerm(reshape_shape, perm);
      const size_t elem_size = ElementSize(elem_type);
      if (elem_size == 0) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "ShapeCastTranspose unsupported dtype");
      }
      if (!IsGpuMemory(data.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "ShapeCastTranspose requires MUSA input");
      }

      Ort::UnownedValue output = ctx.GetOutput(plan.output_index, output_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "ShapeCastTranspose requires MUSA output");
      }

      const bool try_mudnn =
          !IsEnvEnabled("MUSA_EP_DISABLE_MUDNN_SHAPE_CAST_TRANSPOSE");
      if (try_mudnn &&
          TryMudnnTranspose(data, output, reshape_shape, output_shape, perm,
                            elem_type)) {
        continue;
      }

      OrtStatus* status = LaunchStatus(LaunchMusaTransposeKernel(
          data.GetTensorRawData(), output.GetTensorMutableRawData(),
          static_cast<int32_t>(elem_size),
          MakeTransposeParams(reshape_shape, output_shape, perm), nullptr));
      if (status != nullptr) {
        return status;
      }
    }
    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

bool IsShapeCastTransposeFusionGraph(Ort::ConstGraph graph) {
  bool has_concat = false;
  bool has_cast = false;
  bool has_reshape = false;
  bool has_transpose = false;
  for (Ort::ConstNode node : graph.GetNodes()) {
    has_concat = has_concat || IsOnnxOp(node, "Concat");
    has_cast = has_cast || IsOnnxOp(node, "Cast");
    has_reshape = has_reshape || IsOnnxOp(node, "Reshape");
    has_transpose = has_transpose || IsOnnxOp(node, "Transpose");
  }
  return has_concat && has_cast && has_reshape && has_transpose;
}

std::unique_ptr<FusionNodeCompute> CreateShapeCastTransposeFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  return std::make_unique<ShapeCastTransposeFusionCompute>(
      BuildOutputPlans(graph, fused_node));
}
