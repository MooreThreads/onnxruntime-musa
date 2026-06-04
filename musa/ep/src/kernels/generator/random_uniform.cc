// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/random_impl.h"
#include "shared_inc/op_kernel_common.h"

namespace {

class RandomUniformBase {
 protected:
  explicit RandomUniformBase(const OrtKernelInfo* info) {
    Ort::ConstKernelInfo kernel_info(info);
    low_ = AttrOrDefault<float>(kernel_info, "low", 0.0f);
    high_ = AttrOrDefault<float>(kernel_info, "high", 1.0f);
    dtype_ = AttrOrDefault<int64_t>(
        kernel_info, "dtype", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    seed_ = static_cast<uint64_t>(
        AttrOrDefault<float>(kernel_info, "seed", 0.0f));
  }

  OrtStatus* FillOutput(Ort::UnownedValue output,
                        const std::vector<int64_t>& shape,
                        ONNXTensorElementDataType elem_type) const {
    if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "RandomUniform requires MUSA output");
    }
    MusaElementType musa_elem_type;
    if (!ToMusaElementType(elem_type, musa_elem_type)) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "RandomUniform unsupported dtype");
    }
    musaError_t status = LaunchMusaRandomUniformKernel(
        output.GetTensorMutableRawData(), NumElements(shape), low_, high_,
        seed_, musa_elem_type, nullptr);
    if (status == musaErrorNotSupported) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "RandomUniform unsupported dtype");
    }
    return LaunchStatus(status);
  }

  float low_ = 0.0f;
  float high_ = 1.0f;
  int64_t dtype_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  uint64_t seed_ = 0;
};

class RandomUniform : public OpKernelBase<RandomUniform>,
                      private RandomUniformBase {
 public:
  RandomUniform(const OrtKernelInfo* info, void* /*state*/)
      : RandomUniformBase(info) {
    Ort::ConstKernelInfo kernel_info(info);
    shape_ = AttrsOrEmpty(kernel_info, "shape");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const {
    auto elem_type = static_cast<ONNXTensorElementDataType>(dtype_);
    Ort::UnownedValue output = ctx.GetOutput(0, shape_);
    return FillOutput(output, shape_, elem_type);
  }

 private:
  std::vector<int64_t> shape_;
};

class RandomUniformLike : public OpKernelBase<RandomUniformLike>,
                          private RandomUniformBase {
 public:
  RandomUniformLike(const OrtKernelInfo* info, void* /*state*/)
      : RandomUniformBase(info) {
    Ort::ConstKernelInfo kernel_info(info);
    has_dtype_attr_ = HasDType(kernel_info);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const {
    Ort::ConstValue input = ctx.GetInput(0);
    auto input_info = input.GetTensorTypeAndShapeInfo();
    auto output_shape = input_info.GetShape();
    auto elem_type =
        has_dtype_attr_
            ? static_cast<ONNXTensorElementDataType>(dtype_)
            : input_info.GetElementType();
    Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
    return FillOutput(output, output_shape, elem_type);
  }

 private:
  static bool HasDType(Ort::ConstKernelInfo& kernel_info) {
    try {
      (void)kernel_info.GetAttribute<int64_t>("dtype");
      return true;
    } catch (...) {
      return false;
    }
  }

  bool has_dtype_attr_ = false;
};

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    RandomUniform, kOnnxDomain, 1, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    RandomUniform)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    RandomUniformLike, kOnnxDomain, 1, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T1", AllFixedSizeTensorTypes())
         .AddTypeConstraint("T2", FloatLikeTensorTypes())),
    RandomUniformLike)
