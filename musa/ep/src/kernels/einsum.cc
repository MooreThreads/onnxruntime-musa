// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <map>
#include <sstream>

#include "common/op_kernel_common.h"

namespace {
class Einsum : public OpKernelBase<Einsum> {
 public:
  Einsum(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo ki(info);
    equation_ = AttrOrDefault<std::string>(ki, "equation", std::string(""));
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  std::string equation_;
};

OrtStatus* Einsum::Compute(Ort::KernelContext& ctx) const {
  size_t arrow = equation_.find("->");
  if (arrow == std::string::npos)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum: implicit output not supported");

  std::string inputs_spec = equation_.substr(0, arrow);
  std::string output_spec = equation_.substr(arrow + 2);

  // Split inputs_spec by ','
  std::vector<std::string> input_specs;
  {
    std::string token;
    for (char ch : inputs_spec) {
      if (ch == ',') {
        input_specs.push_back(token);
        token.clear();
      } else {
        token += ch;
      }
    }
    input_specs.push_back(token);
  }

  size_t num_inputs = ctx.GetInputCount();
  if (input_specs.size() != num_inputs)
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Einsum: equation/input count mismatch");

  // Check all inputs are float
  for (size_t i = 0; i < num_inputs; ++i) {
    if (ctx.GetInput(i).GetTensorTypeAndShapeInfo().GetElementType() !=
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "Einsum: only float32 supported");
  }

  // Map each label to its dimension size
  std::map<char, int64_t> label_dim;
  for (size_t inp = 0; inp < num_inputs; ++inp) {
    auto shape = ctx.GetInput(inp).GetTensorTypeAndShapeInfo().GetShape();
    const std::string& spec = input_specs[inp];
    for (size_t d = 0; d < spec.size(); ++d) {
      char label = spec[d];
      if (label_dim.find(label) == label_dim.end()) label_dim[label] = shape[d];
    }
  }

  // Build "all_labels" in a deterministic order: output labels first, then
  // contracted
  std::string all_labels;
  for (char c : output_spec)
    if (all_labels.find(c) == std::string::npos) all_labels += c;
  for (auto& [label, _] : label_dim)
    if (all_labels.find(label) == std::string::npos) all_labels += label;

  // Build index maps
  std::map<char, size_t> label_to_all_idx;
  for (size_t i = 0; i < all_labels.size(); ++i)
    label_to_all_idx[all_labels[i]] = i;

  std::vector<int64_t> all_dims(all_labels.size());
  for (char c : all_labels) all_dims[label_to_all_idx[c]] = label_dim[c];

  std::vector<int64_t> out_shape;
  for (char c : output_spec) out_shape.push_back(label_dim[c]);

  // Read input data
  std::vector<std::vector<float>> in_data(num_inputs);
  std::vector<std::vector<int64_t>> in_shapes(num_inputs);
  std::vector<std::vector<int64_t>> in_strides(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    in_data[i] = ReadTyped<float>(ctx.GetInput(i));
    in_shapes[i] = ctx.GetInput(i).GetTensorTypeAndShapeInfo().GetShape();
    in_strides[i] = Strides(in_shapes[i]);
  }

  auto out_strides = Strides(out_shape);
  int64_t total_all = NumElements(all_dims);
  std::vector<float> out_data(static_cast<size_t>(NumElements(out_shape)),
                              0.0f);

  for (int64_t idx = 0; idx < total_all; ++idx) {
    auto coord = Coordinates(idx, all_dims);

    // Compute output index
    std::vector<int64_t> oc;
    for (char c : output_spec) oc.push_back(coord[label_to_all_idx[c]]);
    int64_t o_idx = Offset(oc, out_strides);

    // Compute product of inputs
    float product = 1.0f;
    for (size_t inp = 0; inp < num_inputs; ++inp) {
      std::vector<int64_t> ic;
      for (char c : input_specs[inp]) ic.push_back(coord[label_to_all_idx[c]]);
      int64_t i_idx = Offset(ic, in_strides[inp]);
      product *= in_data[inp][static_cast<size_t>(i_idx)];
    }
    out_data[static_cast<size_t>(o_idx)] += product;
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return WriteTyped<float>(y, out_data);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Einsum, kOnnxDomain, 12, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    Einsum)
