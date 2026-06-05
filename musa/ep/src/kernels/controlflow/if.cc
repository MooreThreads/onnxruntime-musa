// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {

class If {
 public:
  static OrtStatus* CreateKernelImpl(const OrtKernelInfo* info, void* /*state*/,
                                     OrtKernelImpl*& kernel) noexcept {
    EXCEPTION_TO_RETURNED_STATUS_BEGIN
    kernel = nullptr;
    RETURN_IF_ERROR(Ort::GetEpApi().CreateIfKernel(info, &kernel));
    return nullptr;
    EXCEPTION_TO_RETURNED_STATUS_END
  }
};

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    If, kOnnxDomain, 13, 15,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("B",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", AllFixedSizeTensorTypesNoBFloat16())
         .SetInputMemType(0, OrtMemTypeCPUInput)),
    If)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    If, kOnnxDomain, 16, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("B",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", AllFixedSizeTensorTypes())
         .SetInputMemType(0, OrtMemTypeCPUInput)),
    If)
