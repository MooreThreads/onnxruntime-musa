#include "ep_kernel_registration.h"

#include <vector>

#include "kernels/utils.h"

static const BuildKernelCreateInfoFn build_kernel_create_info_funcs[] = {
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, MatMul)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 13, Add)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 14, 19, Add)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 13, Sub)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 14, 19, Sub)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 13, Mul)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 14, 19, Mul)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 13, Div)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 14, 19, Div)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 14, Pow)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 15, 19, Pow)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Sum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Gemm)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kMSDomain, 1, 1, FusedGemm)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kMSDomain, 1, 1, FusedMatMul)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Relu)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 15, LeakyRelu)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 16, 19, LeakyRelu)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Sqrt)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Reciprocal)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Neg)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Log)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Tanh)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Sigmoid)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Softmax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Abs)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Erf)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Equal)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Greater)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Max)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Min)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 1, 19, Not)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 7, 19, Or)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Shape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Cast)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Reshape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Unsqueeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Expand)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Concat)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Transpose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Gather)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Slice)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, Split)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, ReduceProd)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, ReduceSum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, ReduceMean)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 19, ReduceSumSquare)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 15, 17, BatchNormalization)>,
};

size_t GetNumKernels() { return std::size(build_kernel_create_info_funcs); }

static OrtStatus* RegisterKernels(Ort::KernelRegistry& kernel_registry,
                                  const char* ep_name,
                                  void* create_kernel_state) {
  for (auto& build_func : build_kernel_create_info_funcs) {
    KernelCreateInfo kernel_create_info = {};
    RETURN_IF_ERROR(
        build_func(ep_name, create_kernel_state, &kernel_create_info));

    if (kernel_create_info.kernel_def != nullptr) {
      RETURN_IF_ERROR(kernel_registry.AddKernel(
          kernel_create_info.kernel_def, kernel_create_info.kernel_create_func,
          kernel_create_info.kernel_create_func_state));
    }
  }

  return nullptr;
}

OrtStatus* CreateKernelRegistry(const char* ep_name, void* create_kernel_state,
                                OrtKernelRegistry** out_kernel_registry) {
  *out_kernel_registry = nullptr;

  if (GetNumKernels() == 0) {
    return nullptr;
  }

  try {
    Ort::KernelRegistry kernel_registry;
    Ort::Status status{
        RegisterKernels(kernel_registry, ep_name, create_kernel_state)};

    *out_kernel_registry = status.IsOK() ? kernel_registry.release() : nullptr;
    return status.release();
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }
}
