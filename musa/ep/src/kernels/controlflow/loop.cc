// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {

class MusaLoopKernelHelper : public OrtLoopKernelHelper {
 public:
  MusaLoopKernelHelper() {
    ort_version_supported = ORT_API_VERSION;
    Release = ReleaseImpl;
    ConcatOutput = ConcatOutputImpl;
  }

 private:
  static void ORT_API_CALL ReleaseImpl(OrtLoopKernelHelper* this_ptr) noexcept {
    delete static_cast<MusaLoopKernelHelper*>(this_ptr);
  }

  static OrtStatus* ORT_API_CALL
  ConcatOutputImpl(OrtLoopKernelHelper* /*this_ptr*/, void* stream_handle,
                   const OrtValue* const* per_iteration_outputs,
                   size_t num_per_iteration_outputs, void* output,
                   size_t output_size_in_bytes) noexcept {
    EXCEPTION_TO_RETURNED_STATUS_BEGIN
    if (num_per_iteration_outputs == 0) {
      return nullptr;
    }

    size_t copied = 0;
    auto stream = static_cast<musaStream_t>(stream_handle);
    // Loop body execution may produce per-iteration outputs on the runtime
    // default stream. MUSA compute streams are non-blocking, so make the
    // producer/consumer edge explicit before concatenating scan outputs.
    RETURN_IF_ERROR(WaitForDefaultStream(stream));
    auto* dst = static_cast<uint8_t*>(output);
    for (size_t i = 0; i < num_per_iteration_outputs; ++i) {
      Ort::ConstValue value(per_iteration_outputs[i]);
      if (!IsGpuMemory(value.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "Loop scan outputs must be MUSA device tensors");
      }

      const size_t bytes = value.GetTensorSizeInBytes();
      if (copied + bytes > output_size_in_bytes) {
        return Ort::GetApi().CreateStatus(
            ORT_FAIL, "Loop scan output concatenation exceeds output buffer");
      }
      musaError_t status =
          musaMemcpyAsync(dst + copied, value.GetTensorRawData(), bytes,
                          musaMemcpyDeviceToDevice, stream);
      if (status != musaSuccess) {
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
      }
      copied += bytes;
    }

    if (copied != output_size_in_bytes) {
      return Ort::GetApi().CreateStatus(
          ORT_FAIL,
          "Loop scan output concatenation did not fill output buffer");
    }
    return nullptr;
    EXCEPTION_TO_RETURNED_STATUS_END
  }
};

class Loop {
 public:
  static OrtStatus* CreateKernelImpl(const OrtKernelInfo* info, void* /*state*/,
                                     OrtKernelImpl*& kernel) noexcept {
    EXCEPTION_TO_RETURNED_STATUS_BEGIN
    kernel = nullptr;
    auto* helper = new MusaLoopKernelHelper();
    OrtStatus* status = Ort::GetEpApi().CreateLoopKernel(info, helper, &kernel);
    if (status != nullptr) {
      helper->Release(helper);
      return status;
    }
    return nullptr;
    EXCEPTION_TO_RETURNED_STATUS_END
  }
};

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Loop, kOnnxDomain, 1, 12,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("I",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("B",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", AllFixedSizeTensorTypesNoBFloat16())
         .SetInputMemType(0, OrtMemTypeCPUInput)
         .SetInputMemType(1, OrtMemTypeCPUInput)),
    Loop)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Loop, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("I",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("B",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", AllFixedSizeTensorTypes())
         .SetInputMemType(0, OrtMemTypeCPUInput)
         .SetInputMemType(1, OrtMemTypeCPUInput)),
    Loop)
