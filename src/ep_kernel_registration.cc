#include "ep_kernel_registration.h"

#include <array>

#include "kernel_utils.h"
#include "kernels/add.h"
#include "kernels/matmul.h"

namespace {

template <typename KernelClass>
OrtStatus* CreateKernel(void* state, const OrtKernelInfo* info,
                        OrtKernelImpl** kernel_out) noexcept {
  RETURN_IF(kernel_out == nullptr, Ort::GetApi(),
            "CreateKernel received a null output pointer.");
  *kernel_out = nullptr;
  RETURN_IF_ERROR(KernelClass::CreateKernelImpl(info, state, *kernel_out));
  return nullptr;
}

template <typename KernelClass>
OrtStatus* BuildFloatKernelCreateInfo(const char* ep_name, const char* op_type,
                                      int since_version,
                                      void* create_kernel_state,
                                      KernelCreateInfo* result) {
  EXCEPTION_TO_STATUS_BEGIN
  result->kernel_def =
      Ort::KernelDefBuilder()
          .SetOperatorType(op_type)
          .SetDomain("")
          .SetSinceVersion(since_version, since_version)
          .SetExecutionProvider(ep_name)
          .AddTypeConstraint(
              "T", TensorDataType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
          .Build();
  result->kernel_create_func = CreateKernel<KernelClass>;
  result->kernel_create_func_state = create_kernel_state;
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* BuildAddCreateInfo(const char* ep_name, void* state,
                              KernelCreateInfo* result) {
  return BuildFloatKernelCreateInfo<AddKernel>(ep_name, "Add", 14, state,
                                               result);
}

OrtStatus* BuildMatMulCreateInfo(const char* ep_name, void* state,
                                 KernelCreateInfo* result) {
  return BuildFloatKernelCreateInfo<MatMulKernel>(ep_name, "MatMul", 13, state,
                                                  result);
}

constexpr std::array<BuildKernelCreateInfoFn, 2> kKernelBuilders = {
    BuildAddCreateInfo, BuildMatMulCreateInfo};

OrtStatus* RegisterKernels(Ort::KernelRegistry& registry, const char* ep_name,
                           void* state) {
  for (BuildKernelCreateInfoFn builder : kKernelBuilders) {
    KernelCreateInfo create_info;
    RETURN_IF_ERROR(builder(ep_name, state, &create_info));
    RETURN_IF_ERROR(registry.AddKernel(create_info.kernel_def,
                                       create_info.kernel_create_func,
                                       create_info.kernel_create_func_state));
  }
  return nullptr;
}

}  // namespace

size_t GetNumKernels() { return kKernelBuilders.size(); }

OrtStatus* CreateKernelRegistry(const char* ep_name, void* create_kernel_state,
                                OrtKernelRegistry** out_kernel_registry) {
  EXCEPTION_TO_STATUS_BEGIN
  *out_kernel_registry = nullptr;
  Ort::KernelRegistry registry;
  Ort::Status status{RegisterKernels(registry, ep_name, create_kernel_state)};
  *out_kernel_registry = status.IsOK() ? registry.release() : nullptr;
  return status.release();
  EXCEPTION_TO_STATUS_END
}
