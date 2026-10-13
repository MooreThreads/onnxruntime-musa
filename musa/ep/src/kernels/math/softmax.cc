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

#include "math/softmax_impl.h"
#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {
bool IsSoftmaxDeviceType(MusaElementType elem_type) {
  return elem_type == MusaElementType::Float ||
         elem_type == MusaElementType::Double ||
         elem_type == MusaElementType::Float16 ||
         elem_type == MusaElementType::BFloat16;
}

bool TryMudnnSoftmax(Ort::KernelContext& ctx, const std::vector<int64_t>& shape,
                     int64_t axis, ONNXTensorElementDataType elem_type,
                     Ort::UnownedValue y) {
  Ort::ConstValue input = ctx.GetInput(0);
  if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(y.GetTensorMemoryInfo())) {
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
      !SetMudnnTensor(output_tensor, y.GetTensorMutableRawData(), shape,
                      elem_type)) {
    return false;
  }

  ::musa::dnn::Softmax op;
  if (op.SetMode(::musa::dnn::Softmax::Mode::SOFTMAX) !=
          ::musa::dnn::Status::SUCCESS ||
      op.SetAlgorithm(::musa::dnn::Softmax::Algorithm::ACCURATE) !=
          ::musa::dnn::Status::SUCCESS ||
      op.SetDim(static_cast<int>(axis)) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, input_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

class Softmax : public OpKernelBase<Softmax> {
 public:
  Softmax(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    // The axis semantics changed at opset 13: through opset 12 the input is
    // coerced into the 2-D matrix [prod(d0..d_{axis-1}), prod(d_axis..)] and
    // normalised along its second dimension, with axis defaulting to 1; from
    // opset 13 on the normalisation runs along `axis` alone and the default
    // is -1. ORT resolves the node against the ONNX schema of the model's
    // opset, so the node's since version tells the two apart.
    since_version_ = kernel_info.GetOperatorSinceVersion();
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis",
                                   since_version_ >= 13 ? -1 : 1);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = 0;
  int64_t since_version_ = 0;
};

OrtStatus* Softmax::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      !IsSoftmaxDeviceType(musa_elem_type)) {
    return UnsupportedDeviceElementwiseStatus("Softmax", elem_type);
  }
  std::vector<int64_t> shape0 = info.GetShape();
  const bool flatten_trailing_dims = since_version_ < 13;
  int64_t axis = 0;
  int64_t outer = 1;
  int64_t dim = 1;
  int64_t inner = 1;
  // The pre-13 reading of `axis` is the opset-13 one for every node whose
  // trailing dimensions are all one, so most old-opset nodes keep the plain
  // shape/axis pair the muDNN softmax is called with.
  bool same_as_axis_reduction = !flatten_trailing_dims;
  if (!shape0.empty()) {
    const int64_t rank = static_cast<int64_t>(shape0.size());
    axis = NormalizeAxis(axis_, shape0.size());
    // For the flattened form `axis == rank` is meaningful (it reduces a single
    // element, which is what the promoted 2-D matrix [prod(d0..d_{rank-1}), 1]
    // holds), and it is what a 1-D input with the opset-12 default axis hits.
    const int64_t highest_axis = flatten_trailing_dims ? rank : rank - 1;
    if (axis < 0 || axis > highest_axis) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "Softmax axis out of range");
    }
    for (int64_t i = 0; i < axis; ++i) {
      outer *= shape0[static_cast<size_t>(i)];
    }
    if (flatten_trailing_dims) {
      for (int64_t i = axis; i < rank; ++i) {
        dim *= shape0[static_cast<size_t>(i)];
      }
      same_as_axis_reduction = axis < rank &&
                               shape0[static_cast<size_t>(axis)] > 0 &&
                               dim == shape0[static_cast<size_t>(axis)];
    } else {
      dim = shape0[static_cast<size_t>(axis)];
      for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= shape0[static_cast<size_t>(i)];
      }
    }
  }
  // muDNN gets the contiguous view whose last dimension is the one to reduce:
  // the node's own shape when `axis` already selects it, and the promoted
  // [prod(d0..d_{axis-1}), prod(d_axis..)] matrix of the pre-13 schemas
  // otherwise. Both describe exactly the reduction the node asks for, and a
  // muDNN rejection falls through to the device kernel below.
  std::vector<int64_t> reduction_shape = shape0;
  int64_t reduction_axis = axis;
  if (!same_as_axis_reduction) {
    reduction_shape = {outer, dim};
    reduction_axis = 1;
  }
  Ort::UnownedValue y = ctx.GetOutput(0, shape0);
  if (TryMudnnSoftmax(ctx, reduction_shape, reduction_axis, elem_type, y)) {
    return nullptr;
  }

  if (!IsGpuMemory(input0.GetTensorMemoryInfo()) ||
      !IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus("Softmax", elem_type);
  }
  return LaunchStatus(LaunchMusaSoftmaxKernel(
      input0.GetTensorRawData(), y.GetTensorMutableRawData(), outer, dim, inner,
      musa_elem_type, GetComputeStream(ctx)));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Softmax, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    Softmax)

// The pre-13 schemas do not have bfloat16; opset 11-12 only clarified that a
// negative axis counts from the back, so one class serves both old ranges and
// picks the semantics from the node's since version.
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Softmax, kOnnxDomain, 1, 10,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())), Softmax)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Softmax, kOnnxDomain, 11, 12,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())), Softmax)
