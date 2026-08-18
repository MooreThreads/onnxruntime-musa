// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "fusion/silu_fusion.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "graph/graph_utils.h"
#include "kernels/activation/silu_impl.h"
#include "kernels/shared_inc/blas_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"

namespace musa_ep {
namespace {

bool FindSiluGraphInput(Ort::ConstGraph graph, std::string& input_name) {
  Ort::ConstNode sigmoid{nullptr};
  Ort::ConstNode mul{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Sigmoid")) {
      if (sigmoid) {
        return false;
      }
      sigmoid = node;
    } else if (IsOnnxOp(node, "Mul")) {
      if (mul) {
        return false;
      }
      mul = node;
    } else {
      return false;
    }
  }
  if (!sigmoid || !mul) {
    return false;
  }

  const auto sigmoid_inputs = sigmoid.GetInputs();
  const auto sigmoid_outputs = sigmoid.GetOutputs();
  const auto mul_inputs = mul.GetInputs();
  if (sigmoid_inputs.size() != 1 || sigmoid_outputs.size() != 1 ||
      mul_inputs.size() != 2) {
    return false;
  }

  const std::string x_name = Name(sigmoid_inputs[0]);
  const std::string sigmoid_output_name = Name(sigmoid_outputs[0]);
  const std::string lhs_name = Name(mul_inputs[0]);
  const std::string rhs_name = Name(mul_inputs[1]);
  const bool lhs_is_sigmoid = lhs_name == sigmoid_output_name;
  const bool rhs_is_sigmoid = rhs_name == sigmoid_output_name;
  if (lhs_is_sigmoid == rhs_is_sigmoid) {
    return false;
  }
  const std::string other_name = lhs_is_sigmoid ? rhs_name : lhs_name;
  if (other_name != x_name) {
    return false;
  }

  input_name = x_name;
  return true;
}

bool TryMudnnSilu(Ort::KernelContext& ctx, Ort::ConstValue input,
                  Ort::UnownedValue output, const std::vector<int64_t>& shape,
                  ONNXTensorElementDataType elem_type) {
  constexpr size_t kMudnnMaxElementwiseRank = 5;
  if (shape.size() > kMudnnMaxElementwiseRank ||
      !IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, GetComputeStream(ctx));
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor input_tensor;
  ::musa::dnn::Tensor output_tensor;
  if (!SetMudnnTensor(input_tensor, input.GetTensorRawData(), shape,
                      elem_type) ||
      !SetMudnnTensor(output_tensor, output.GetTensorMutableRawData(), shape,
                      elem_type)) {
    return false;
  }

  ::musa::dnn::Unary op;
  if (op.SetMode(::musa::dnn::Unary::Mode::SILU) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, input_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

}  // namespace

OrtStatus* SiluFusionCompute::Compute(OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    Ort::ConstValue input = ctx.GetInput(input_index);
    const auto input_info = input.GetTensorTypeAndShapeInfo();
    const auto elem_type = input_info.GetElementType();
    const auto shape = input_info.GetShape();
    if (OutputEmptyTensorIfNeeded(ctx, shape)) {
      return nullptr;
    }

    Ort::UnownedValue output = ctx.GetOutput(0, shape);
    if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
        !IsGpuMemory(output.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "Silu fusion requires MUSA input/output");
    }

    if (TryMudnnSilu(ctx, input, output, shape, elem_type)) {
      return nullptr;
    }

    MusaElementType musa_elem_type;
    if (!ToMusaElementType(elem_type, musa_elem_type)) {
      return UnsupportedDeviceElementwiseStatus("SiluFusion", elem_type);
    }
    return LaunchStatus(LaunchMusaSiluKernel(
        input.GetTensorRawData(), output.GetTensorMutableRawData(),
        NumElements(shape), musa_elem_type, GetComputeStream(ctx)));
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

bool IsSiluFusionGraph(Ort::ConstGraph graph) {
  std::string input_name;
  return FindSiluGraphInput(graph, input_name);
}

std::unique_ptr<FusionNodeCompute> CreateSiluFusion(Ort::ConstGraph graph,
                                                    Ort::ConstNode fused_node) {
  std::string input_name;
  if (!FindSiluGraphInput(graph, input_name)) {
    throw std::runtime_error("invalid Silu fusion graph");
  }

  const auto inputs = fused_node.GetInputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (Name(inputs[i]) == input_name) {
      return std::make_unique<SiluFusionCompute>(i);
    }
  }
  throw std::runtime_error("Silu fusion input missing");
}

}  // namespace musa_ep
