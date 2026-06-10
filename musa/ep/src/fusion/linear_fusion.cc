// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/linear_fusion.h"

#include <mublas.h>
#include <musa_runtime.h>

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/shared_inc/blas_utils.h"

/*
 * Linear Fusion Patterns
 *
 * Pattern 1: Gemm + activation
 *
 *   Gemm
 *     inputs: A, B, optional C
 *     attrs: alpha, beta, transA, transB
 *      |
 *      v
 *   Relu | LeakyRelu | Tanh
 *
 *   Semantics:
 *     Y = activation(alpha * Gemm(A, B) + beta * C)
 *
 * Pattern 2: MatMul + Add + activation
 *
 *   MatMul(A, B)
 *      |
 *      v
 *   Add(..., bias)
 *      |
 *      v
 *   Relu | LeakyRelu | Tanh
 *
 *   Semantics:
 *     Y = activation(MatMul(A, B) + bias)
 *
 * Runtime path:
 *   Both patterns are lowered to the shared MUSA GEMM helper so bias and
 *   activation can run on device without materializing unnecessary host-side
 *   intermediates.
 */

namespace {

constexpr size_t kNoBiasInput = std::numeric_limits<size_t>::max();

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(const Ort::ConstNode& node, const char* op_type) {
  return node.GetOperatorType() == op_type && IsOnnxDomain(node.GetDomain());
}

bool IsActivationOp(const Ort::ConstNode& node) {
  return IsOnnxOp(node, "Relu") || IsOnnxOp(node, "LeakyRelu") ||
         IsOnnxOp(node, "Tanh") || IsOnnxOp(node, "Sigmoid");
}

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
}

std::vector<int64_t> TensorShape(Ort::ConstValue value) {
  return value.GetTensorTypeAndShapeInfo().GetShape();
}

int64_t NumElementsChecked(const std::vector<int64_t>& shape) {
  int64_t total = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error("FusedGemm requires positive runtime shapes");
    }
    if (total > INT64_MAX / dim) {
      throw std::runtime_error("FusedGemm shape overflows int64");
    }
    total *= dim;
  }
  return total;
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

float ReadFloatAttribute(Ort::ConstNode node, const std::string& name,
                         float default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK()) {
    return default_value;
  }

  float value = default_value;
  status = attr.GetValue(value);
  return status.IsOK() ? value : default_value;
}

void ValidateFloatTensor(Ort::ConstValue value, const char* name) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(
        std::string("FusedGemm only supports float tensors for ") + name);
  }
}

std::vector<int64_t> BiasShapeForOutput(
    const std::vector<int64_t>& c_shape, const std::vector<int64_t>& y_shape,
    const std::vector<int64_t>& flat_y_shape) {
  if (BroadcastShape(c_shape, y_shape) == y_shape) {
    if (c_shape.size() == 1 || (c_shape.size() == 2 && c_shape[0] == 1) ||
        c_shape == flat_y_shape) {
      return c_shape;
    }
  }
  if (BroadcastShape(c_shape, flat_y_shape) == flat_y_shape) {
    return c_shape;
  }
  throw std::runtime_error("FusedGemm bias broadcast shape mismatch");
}

OrtStatus* RunDeviceFusedGemm(
    float* y_data, const float* a_data, const float* b_data,
    const float* c_data, const std::vector<int64_t>& a_shape,
    const std::vector<int64_t>& b_shape, const std::vector<int64_t>& c_shape,
    const std::vector<int64_t>& y_shape, bool trans_a, bool trans_b,
    float alpha, float beta, const std::string& activation,
    float activation_alpha, bool has_bias, musaStream_t stream) {
  GemmShapeInfo shape_info;
  RETURN_IF_ERROR(ResolveGemmShape(a_shape, b_shape, trans_a, trans_b,
                                   shape_info));
  const int64_t m = shape_info.m;
  const int64_t k = shape_info.k;
  const int64_t n = shape_info.n;
  if (m > INT32_MAX || k > INT32_MAX || n > INT32_MAX) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "FusedGemm dimensions exceed int32 limits");
  }
  if (m == 0 || n == 0) {
    return nullptr;
  }

  if (TryMudnnGemm(y_data, a_data, b_data, c_data, a_shape, b_shape,
                   c_shape, y_shape, trans_a, trans_b, alpha, beta, has_bias,
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, stream)) {
    if (activation.empty()) {
      return nullptr;
    }

    MusaUnaryOp activation_op = MusaUnaryOp::Relu;
    if (!TryMapActivation(activation, activation_op)) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "unsupported FusedGemm activation");
    }
    MusaBroadcastParams params = MakeBroadcastParams(y_shape, y_shape, {1});
    return LaunchStatus(LaunchMusaGemmPostFloatKernel(
        y_data, nullptr, params, false, 0.0f, activation_op, true,
        activation_alpha, stream));
  }

  mublasHandle_t handle = nullptr;
  RETURN_IF_ERROR(EnsureMublasHandle(&handle, stream));
  mublasOperation_t op_a = trans_a ? MUBLAS_OP_T : MUBLAS_OP_N;
  mublasOperation_t op_b = trans_b ? MUBLAS_OP_T : MUBLAS_OP_N;
  int lda = static_cast<int>(shape_info.lda);
  int ldb = static_cast<int>(shape_info.ldb);
  int mi = static_cast<int>(m);
  int ki = static_cast<int>(k);
  int ni = static_cast<int>(n);
  mublasStatus status =
      MublasGemmEx(handle, op_b, op_a, ni, mi, ki, alpha, b_data, ldb,
                   a_data, lda, 0.0, y_data, ni,
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  if (status != MUBLAS_STATUS_SUCCESS) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "mublasSgemm failed");
  }

  MusaUnaryOp activation_op = MusaUnaryOp::Relu;
  const bool has_activation = !activation.empty();
  if (has_activation && !TryMapActivation(activation, activation_op)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "unsupported FusedGemm activation");
  }

  MusaBroadcastParams params = MakeBroadcastParams(y_shape, y_shape, c_shape);
  return LaunchStatus(LaunchMusaGemmPostFloatKernel(
      y_data, c_data, params, has_bias, beta, activation_op, has_activation,
      activation_alpha, stream));
}

struct LinearFusionCompute : FusionNodeCompute {
  LinearFusionCompute(size_t a_input_index, size_t b_input_index,
                      size_t bias_input_index, bool trans_a, bool trans_b,
                      float alpha, float beta, bool flatten_a,
                      std::string activation, float activation_alpha)
      : a_input_index(a_input_index),
        b_input_index(b_input_index),
        bias_input_index(bias_input_index),
        trans_a(trans_a),
        trans_b(trans_b),
        alpha(alpha),
        beta(beta),
        flatten_a(flatten_a),
        activation(std::move(activation)),
        activation_alpha(activation_alpha) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue a = ctx.GetInput(a_input_index);
      Ort::ConstValue b = ctx.GetInput(b_input_index);
      musaStream_t stream = GetComputeStream(ctx);
      ValidateFloatTensor(a, "A");
      ValidateFloatTensor(b, "B");

      std::vector<int64_t> a_shape = TensorShape(a);
      std::vector<int64_t> b_shape = TensorShape(b);
      std::vector<int64_t> c_shape = {1};
      const float* c_data = nullptr;
      bool has_bias = bias_input_index != kNoBiasInput;
      Ort::ConstValue c{nullptr};
      if (has_bias) {
        c = ctx.GetInput(bias_input_index);
        ValidateFloatTensor(c, "bias");
        c_shape = TensorShape(c);
        c_data = c.GetTensorData<float>();
      }

      if (b_shape.size() != 2) {
        return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                          "FusedGemm requires B rank 2");
      }

      std::vector<int64_t> compute_a_shape = a_shape;
      std::vector<int64_t> y_shape;
      if (flatten_a) {
        if (trans_a || a_shape.size() < 2) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED,
              "FusedGemm flatten path requires non-transposed A rank >= 2");
        }
        const int64_t k = a_shape.back();
        if (k != b_shape[0]) {
          return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                            "FusedGemm K dimension mismatch");
        }
        const int64_t m = NumElementsChecked(a_shape) / k;
        compute_a_shape = {m, k};
        y_shape = a_shape;
        y_shape.back() = trans_b ? b_shape[0] : b_shape[1];
      } else {
        GemmShapeInfo gemm_shape;
        RETURN_IF_ERROR(ResolveGemmShape(a_shape, b_shape, trans_a, trans_b,
                                         gemm_shape));
        y_shape = gemm_shape.out_shape;
      }

      if (has_bias) {
        c_shape = BiasShapeForOutput(c_shape, y_shape, y_shape);
      }

      Ort::UnownedValue y = ctx.GetOutput(0, y_shape);
      if (IsGpuMemory(a.GetTensorMemoryInfo()) &&
          IsGpuMemory(b.GetTensorMemoryInfo()) &&
          (!has_bias || IsGpuMemory(c.GetTensorMemoryInfo())) &&
          IsGpuMemory(y.GetTensorMemoryInfo())) {
        return RunDeviceFusedGemm(
            y.GetTensorMutableData<float>(), a.GetTensorData<float>(),
            b.GetTensorData<float>(), c_data, compute_a_shape, b_shape, c_shape,
            y_shape, trans_a, trans_b, alpha, beta, activation,
            activation_alpha, has_bias, stream);
      }

      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "Linear fusion path requires MUSA tensors for all inputs");
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t a_input_index;
  size_t b_input_index;
  size_t bias_input_index;
  bool trans_a;
  bool trans_b;
  float alpha;
  float beta;
  bool flatten_a;
  std::string activation;
  float activation_alpha;
};

std::string ActivationNameAndAlpha(Ort::ConstNode activation_node,
                                   float& activation_alpha) {
  std::string activation = activation_node.GetOperatorType();
  activation_alpha = 0.01f;
  if (activation == "LeakyRelu") {
    activation_alpha = ReadFloatAttribute(activation_node, "alpha", 0.01f);
  }
  return activation;
}

size_t GetFusedInputIndex(
    const std::unordered_map<std::string, size_t>& fused_input_indices,
    const std::string& input_name) {
  auto it = fused_input_indices.find(input_name);
  if (it == fused_input_indices.end()) {
    throw std::runtime_error("unable to map FusedGemm input " + input_name);
  }
  return it->second;
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

std::unique_ptr<FusionNodeCompute> CreateGemmActivationFusion(
    Ort::ConstNode gemm_node, Ort::ConstNode activation_node,
    Ort::ConstNode fused_node) {
  std::vector<Ort::ConstValueInfo> gemm_inputs = gemm_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gemm_outputs = gemm_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> activation_inputs =
      activation_node.GetInputs();
  if ((gemm_inputs.size() != 2 && gemm_inputs.size() != 3) ||
      gemm_outputs.size() != 1 || activation_inputs.size() != 1 ||
      Name(gemm_outputs[0]) != Name(activation_inputs[0])) {
    throw std::runtime_error("invalid GemmActivation fused graph");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  float activation_alpha = 0.01f;
  std::string activation =
      ActivationNameAndAlpha(activation_node, activation_alpha);
  size_t bias_index = kNoBiasInput;
  if (gemm_inputs.size() == 3) {
    bias_index = GetFusedInputIndex(fused_input_indices, Name(gemm_inputs[2]));
  }
  return std::make_unique<LinearFusionCompute>(
      GetFusedInputIndex(fused_input_indices, Name(gemm_inputs[0])),
      GetFusedInputIndex(fused_input_indices, Name(gemm_inputs[1])), bias_index,
      ReadIntAttribute(gemm_node, "transA", 0) != 0,
      ReadIntAttribute(gemm_node, "transB", 0) != 0,
      ReadFloatAttribute(gemm_node, "alpha", 1.0f),
      ReadFloatAttribute(gemm_node, "beta", 1.0f), false, std::move(activation),
      activation_alpha);
}

std::unique_ptr<FusionNodeCompute> CreateMatMulAddActivationFusion(
    Ort::ConstNode matmul_node, Ort::ConstNode add_node,
    Ort::ConstNode activation_node, Ort::ConstNode fused_node) {
  std::vector<Ort::ConstValueInfo> matmul_inputs = matmul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> matmul_outputs = matmul_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> activation_inputs =
      activation_node ? activation_node.GetInputs()
                      : std::vector<Ort::ConstValueInfo>{};
  if (matmul_inputs.size() != 2 || matmul_outputs.size() != 1 ||
      add_inputs.size() != 2 || add_outputs.size() != 1 ||
      (activation_node && activation_inputs.size() != 1)) {
    throw std::runtime_error("invalid MatMulAdd fused graph");
  }

  const std::string matmul_output_name = Name(matmul_outputs[0]);
  int bias_input_idx = -1;
  if (Name(add_inputs[0]) == matmul_output_name) {
    bias_input_idx = 1;
  } else if (Name(add_inputs[1]) == matmul_output_name) {
    bias_input_idx = 0;
  } else {
    throw std::runtime_error("MatMul output does not feed Add");
  }
  if (activation_node && Name(activation_inputs[0]) != Name(add_outputs[0])) {
    throw std::runtime_error("Add output does not feed activation");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  float activation_alpha = 0.01f;
  std::string activation;
  if (activation_node) {
    activation = ActivationNameAndAlpha(activation_node, activation_alpha);
  }
  return std::make_unique<LinearFusionCompute>(
      GetFusedInputIndex(fused_input_indices, Name(matmul_inputs[0])),
      GetFusedInputIndex(fused_input_indices, Name(matmul_inputs[1])),
      GetFusedInputIndex(fused_input_indices,
                         Name(add_inputs[static_cast<size_t>(bias_input_idx)])),
      false, false, 1.0f, 1.0f, true, std::move(activation), activation_alpha);
}

}  // namespace

bool IsLinearFusionGraph(Ort::ConstGraph graph) {
  bool has_gemm = false;
  bool has_matmul = false;
  bool has_add = false;
  bool has_activation = false;
  for (Ort::ConstNode node : graph.GetNodes()) {
    has_gemm = has_gemm || IsOnnxOp(node, "Gemm");
    has_matmul = has_matmul || IsOnnxOp(node, "MatMul");
    has_add = has_add || IsOnnxOp(node, "Add");
    has_activation = has_activation || IsActivationOp(node);
  }
  return (has_gemm && has_activation) ||
         (has_matmul && has_add);
}

std::unique_ptr<FusionNodeCompute> CreateLinearFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode gemm_node{nullptr};
  Ort::ConstNode matmul_node{nullptr};
  Ort::ConstNode add_node{nullptr};
  Ort::ConstNode activation_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Gemm")) {
      gemm_node = node;
    } else if (IsOnnxOp(node, "MatMul")) {
      matmul_node = node;
    } else if (IsOnnxOp(node, "Add")) {
      add_node = node;
    } else if (IsActivationOp(node)) {
      activation_node = node;
    }
  }

  if (gemm_node && activation_node) {
    return CreateGemmActivationFusion(gemm_node, activation_node, fused_node);
  }
  if (matmul_node && add_node) {
    return CreateMatMulAddActivationFusion(matmul_node, add_node, activation_node,
                                      fused_node);
  }

  throw std::runtime_error(
      "FusedGemm fusion expects Gemm+activation or MatMul+Add(+activation)");
}
