#include "ep_kernel_registration.h"

#include <vector>

#include "kernels/utils.h"

static const BuildKernelCreateInfoFn build_kernel_create_info_funcs[] = {
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, MatMul)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Add)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Sub)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Mul)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Div)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Pow)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Sum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Gemm)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kMSDomain, 1, 1, FusedGemm)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kMSDomain, 1, 1, FusedMatMul)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Relu)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, LeakyRelu)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Sqrt)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Reciprocal)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Neg)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Log)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Tanh)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Sigmoid)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Softmax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Abs)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Erf)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Equal)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Greater)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Max)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Min)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 1, 17, Not)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 7, 17, Or)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Shape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Cast)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Reshape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Unsqueeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Expand)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Concat)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Transpose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Gather)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Slice)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Split)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, ReduceProd)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, ReduceSum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, ReduceMean)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, ReduceSumSquare)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 15, 17, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 7, 17, And)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 18, 18, BitwiseAnd)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 9, 17, ConstantOfShape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 11, 17, Conv)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 12, 17, Einsum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Exp)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 12, 17, GreaterOrEqual)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, IsNaN)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Less)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, NonZero)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Pad)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 1, 17, RandomUniform)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 1, 17, RandomUniformLike)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, ReduceMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 11, 17, Round)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 13, 17, Sign)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 1, 17, Softplus)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 6, 17, Tile)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        kOnnxDomain, 9, 17, Where)>,
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
