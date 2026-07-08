// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "../plugin_ep_utils.h"
#include "runtime_graph_dump.h"

/// <summary>
/// Gets an OrtDataType for a tensor type. Throws on error.
/// </summary>
/// <param name="elem_type"></param>
/// <returns></returns>
inline const OrtDataType* GetTensorType(ONNXTensorElementDataType elem_type) {
  const OrtEpApi& ep_api = Ort::GetEpApi();
  const OrtDataType* result = nullptr;

  Ort::ThrowOnError(ep_api.GetTensorDataType(elem_type, &result));
  return result;
}

/// <summary>
/// Copy a tensor using a OrtDataTransferImpl instance. Used by kernel
/// implementations to copy tensors that my reside on different devices.
/// </summary>
/// <param name="data_transfer_impl"></param>
/// <param name="src_tensor"></param>
/// <param name="dst_tensor"></param>
/// <returns></returns>
inline OrtStatus* CopyTensor(OrtDataTransferImpl& data_transfer_impl,
                             Ort::ConstValue src_tensor,
                             Ort::UnownedValue dst_tensor) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  const OrtMemoryDevice* src_device =
      Ort::GetEpApi().MemoryInfo_GetMemoryDevice(
          src_tensor.GetTensorMemoryInfo());
  const OrtMemoryDevice* dst_device =
      Ort::GetEpApi().MemoryInfo_GetMemoryDevice(
          dst_tensor.GetTensorMemoryInfo());

  RETURN_IF(
      !data_transfer_impl.CanCopy(&data_transfer_impl, src_device, dst_device),
      Ort::GetApi(),
      "OrtDataTransferImpl cannot copy src tensor to dst tensor.");

  auto src_type_shape = src_tensor.GetTensorTypeAndShapeInfo();
  auto dst_type_shape = dst_tensor.GetTensorTypeAndShapeInfo();
  bool same_elem_type =
      src_type_shape.GetElementType() == dst_type_shape.GetElementType();
  bool same_elem_count =
      src_type_shape.GetElementCount() == dst_type_shape.GetElementCount();
  RETURN_IF(!same_elem_type || !same_elem_count, Ort::GetApi(),
            "Cannot copy tensors of different types or size.");

  std::array<const OrtValue*, 1> src_tensors = {src_tensor};
  std::array<OrtValue*, 1> dst_tensors = {dst_tensor};

  RETURN_IF_ERROR(data_transfer_impl.CopyTensors(
      &data_transfer_impl, src_tensors.data(), dst_tensors.data(),
      /*streams*/ nullptr, src_tensors.size()));

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/// <summary>
/// Contains information to create a kernel: kernel definition, creation
/// function + state.
/// </summary>
struct KernelCreateInfo {
  KernelCreateInfo() = default;
  KernelCreateInfo(Ort::KernelDef def, OrtKernelCreateFunc func, void* state)
      : kernel_def{std::move(def)},
        kernel_create_func{func},
        kernel_create_func_state{state} {}

  Ort::KernelDef kernel_def{nullptr};
  OrtKernelCreateFunc kernel_create_func = nullptr;
  void* kernel_create_func_state = nullptr;
};

using BuildKernelCreateInfoFn = OrtStatus* (*)(const char*, void*,
                                               KernelCreateInfo*);

inline std::vector<BuildKernelCreateInfoFn>& RegisteredKernelCreateInfoFuncs() {
  static std::vector<BuildKernelCreateInfoFn> funcs;
  return funcs;
}

class KernelCreateInfoRegistrar {
 public:
  explicit KernelCreateInfoRegistrar(BuildKernelCreateInfoFn build_func) {
    RegisteredKernelCreateInfoFuncs().push_back(build_func);
  }
};

template <typename T>
OrtStatus* BuildKernelCreateInfo(const char* ep_name, void* create_func_state,
                                 /*out*/ KernelCreateInfo* result);

template <>
inline OrtStatus* BuildKernelCreateInfo<void>(
    const char* /*ep_name*/, void* /*create_func_state*/,
    /*out*/ KernelCreateInfo* result) {
  result->kernel_def = Ort::KernelDef{nullptr};
  result->kernel_create_func = nullptr;
  result->kernel_create_func_state = nullptr;
  return nullptr;
}

static constexpr const char* kOnnxDomain = "";
static constexpr const char* kMSDomain = "com.microsoft";

// Naming convention for operator kernel classes with a start and end version
// range.
#define ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, startver, endver, \
                                                  name)                     \
  example_ep_##name##_##domain##_ver##startver##_##endver

// Naming convention for operator kernel classes for a single version
#define ONNX_OPERATOR_KERNEL_CLASS_NAME(domain, version, name) \
  ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, version, version, name)

#define MUSA_CONCAT_IMPL(a, b) a##b
#define MUSA_CONCAT(a, b) MUSA_CONCAT_IMPL(a, b)

// Defines a function of type BuildKernelCreateInfoFn for a kernel
// implementation with a start and end version range.
#define ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, startver, endver,      \
                                          builder, kernel_class)               \
  class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, startver, endver,    \
                                                  name);                       \
  template <>                                                                  \
  OrtStatus* BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(  \
      domain, startver, endver, name)>(const char* ep_name,                    \
                                       void* create_kernel_state,              \
                                       KernelCreateInfo* result) {             \
    try {                                                                      \
      Ort::KernelDef kernel_def = builder.SetOperatorType(#name)               \
                                      .SetDomain(domain)                       \
                                      .SetSinceVersion(startver, endver)       \
                                      .SetExecutionProvider(ep_name)           \
                                      .Build();                                \
                                                                               \
      auto kernel_create_func =                                                \
          [](void* state, const OrtKernelInfo* info,                           \
             OrtKernelImpl** kernel_out) noexcept -> OrtStatus* {              \
        EXCEPTION_TO_RETURNED_STATUS_BEGIN                                     \
        RETURN_IF(kernel_out == nullptr, Ort::GetApi(),                        \
                  "OrtKernelCreateFunc received a NULL kernel_out argument");  \
                                                                               \
        *kernel_out = nullptr;                                                 \
        const bool trace_kernels =                                             \
            std::getenv("MUSA_EP_TRACE_KERNELS") != nullptr;                   \
        const bool dump_runtime_graph = RuntimeGraphDumpEnabled();             \
        std::string node_name;                                                 \
        if (trace_kernels || dump_runtime_graph) {                             \
          Ort::ConstKernelInfo kernel_info(info);                              \
          node_name = kernel_info.GetNodeName();                               \
        }                                                                      \
        if (trace_kernels) {                                                   \
          std::fprintf(stderr, "MUSA_KERNEL_CREATE %s node=%s\n", #name,       \
                       node_name.c_str());                                     \
          std::fflush(stderr);                                                 \
        }                                                                      \
        RETURN_IF_ERROR(                                                       \
            kernel_class::CreateKernelImpl(info, state, *kernel_out));         \
        if (dump_runtime_graph) {                                              \
          Ort::ConstKernelInfo kernel_info(info);                              \
          RuntimeGraphNodeMetadata runtime_node;                               \
          runtime_node.kind = "kernel";                                        \
          runtime_node.display_type = #name;                                   \
          runtime_node.node_name = node_name;                                  \
          runtime_node.domain_name = kernel_info.GetOperatorDomain();          \
          runtime_node.since_version = kernel_info.GetOperatorSinceVersion();  \
          for (size_t i = 0; i < kernel_info.GetInputCount(); ++i) {           \
            runtime_node.inputs.push_back(kernel_info.GetInputName(i));        \
          }                                                                    \
          for (size_t i = 0; i < kernel_info.GetOutputCount(); ++i) {          \
            runtime_node.outputs.push_back(kernel_info.GetOutputName(i));      \
          }                                                                    \
          RegisterRuntimeKernelInstance(*kernel_out, std::move(runtime_node)); \
        }                                                                      \
        if (trace_kernels) {                                                   \
          std::fprintf(stderr, "MUSA_KERNEL_CREATED %s node=%s impl=%p\n",     \
                       #name, node_name.c_str(),                               \
                       static_cast<void*>(*kernel_out));                       \
          std::fflush(stderr);                                                 \
        }                                                                      \
        return nullptr;                                                        \
        EXCEPTION_TO_RETURNED_STATUS_END                                       \
      };                                                                       \
                                                                               \
      *result = KernelCreateInfo(std::move(kernel_def), kernel_create_func,    \
                                 create_kernel_state);                         \
    } catch (const Ort::Exception& ex) {                                       \
      Ort::Status status(ex);                                                  \
      return status.release();                                                 \
    } catch (const std::exception& ex) {                                       \
      Ort::Status status(ex.what(), ORT_EP_FAIL);                              \
      return status.release();                                                 \
    }                                                                          \
    return nullptr;                                                            \
  }                                                                            \
  static const KernelCreateInfoRegistrar MUSA_CONCAT(kernel_registrar_,        \
                                                     __LINE__)(                \
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(         \
          domain, startver, endver, name)>);

// Defines a function of type BuildKernelCreateInfoFn for a kernel
// implementation with a start version.
#define ONNX_OPERATOR_KERNEL_EX(name, domain, version, builder, kernel_class) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, version, version, builder,  \
                                    kernel_class)
