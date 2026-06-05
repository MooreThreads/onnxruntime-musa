// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "math/binary_elementwise_ops_impl.h"
#include "math/unary_elementwise_ops_impl.h"
#include "runtime/ep_musa_utils.h"
#include "utils.h"

// CRTP base shared by every elementary kernel. Derived classes only need to
// provide:
//   - a constructor `Derived(const OrtKernelInfo* info, void* state)` that
//     reads the attributes it cares about, and
//   - `OrtStatus* Compute(Ort::KernelContext& ctx) const`.
template <typename Derived>
class OpKernelBase : public OrtKernelImpl {
 public:
  static OrtStatus* CreateKernelImpl(const OrtKernelInfo* info, void* state,
                                     /*out*/ OrtKernelImpl*& kernel) noexcept {
    EXCEPTION_TO_RETURNED_STATUS_BEGIN
    auto k = std::make_unique<Derived>(info, state);
    kernel = k.release();
    return nullptr;
    EXCEPTION_TO_RETURNED_STATUS_END
  }

  static OrtStatus* ORT_API_CALL
  ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
    EXCEPTION_TO_RETURNED_STATUS_BEGIN
    auto* k = static_cast<Derived*>(this_ptr);
    Ort::KernelContext ctx(kernel_ctx);
    return k->Compute(ctx);
    EXCEPTION_TO_RETURNED_STATUS_END
  }

  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<Derived*>(this_ptr);
  }

 protected:
  OpKernelBase() : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
  }
};

// ---------------------------------------------------------------------------
// Type-constraint helpers. The names are also parsed by
// scripts/gen_supported_ops.py to render musa/docs/supported_ops.md.
// ---------------------------------------------------------------------------
inline std::vector<const OrtDataType*> AllTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

inline std::vector<const OrtDataType*> TensorTypesWithBool() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
  };
}

inline std::vector<const OrtDataType*> AllFixedSizeTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
  };
}

inline std::vector<const OrtDataType*> AllFixedSizeTensorTypesNoBFloat16() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
  };
}

inline std::vector<const OrtDataType*> BinaryNumericOpset13TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> BinaryNumericOpset14TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> PowTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> PowExponentOpset13TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> VariadicMinMaxTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> FloatLikeTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> FloatTensorTypes() {
  return {GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)};
}

inline std::vector<const OrtDataType*> ClipTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
  };
}

inline std::vector<const OrtDataType*> RangeTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> ReduceMeanTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
  };
}

inline std::vector<const OrtDataType*> ReduceL2Opset13TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
  };
}

inline std::vector<const OrtDataType*> HfdTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> AbsTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> NegTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> SignTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> BitwiseIntegerTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

inline std::vector<const OrtDataType*> ReduceMaxTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

inline std::vector<const OrtDataType*> CompareTensorTypesNoBFloat16() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> CompareTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> EqualTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
  };
}

inline std::vector<const OrtDataType*> FloatBoolTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> IntTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

inline std::vector<const OrtDataType*> BoolTensorTypes() {
  return {GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)};
}

inline std::vector<const OrtDataType*> StringTensorTypes() {
  return {GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)};
}

inline std::vector<const OrtDataType*> NonZeroTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
  };
}

inline std::vector<const OrtDataType*> WhereOpset9TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
  };
}

// ---------------------------------------------------------------------------
// Attribute helpers.
// ---------------------------------------------------------------------------
template <typename T>
T AttrOrDefault(Ort::ConstKernelInfo& info, const char* name, T default_value) {
  try {
    return info.GetAttribute<T>(name);
  } catch (...) {
    return default_value;
  }
}

inline std::vector<int64_t> AttrsOrEmpty(Ort::ConstKernelInfo& info,
                                         const char* name) {
  try {
    return info.GetAttributes<int64_t>(name);
  } catch (...) {
    return {};
  }
}

// ---------------------------------------------------------------------------
// Tensor / memory helpers.
// ---------------------------------------------------------------------------
inline bool IsGpuMemory(const OrtMemoryInfo* memory_info) {
  const OrtMemoryDevice* device =
      Ort::GetEpApi().MemoryInfo_GetMemoryDevice(memory_info);
  return Ort::GetEpApi().MemoryDevice_GetDeviceType(device) ==
         OrtMemoryInfoDeviceType_GPU;
}

class DeviceInputBuffer {
 public:
  DeviceInputBuffer() = default;
  ~DeviceInputBuffer() {
    if (ptr_ != nullptr) {
      (void)musaDeviceSynchronize();
      (void)musaFree(ptr_);
    }
  }

  DeviceInputBuffer(const DeviceInputBuffer&) = delete;
  DeviceInputBuffer& operator=(const DeviceInputBuffer&) = delete;

  OrtStatus* Bind(Ort::ConstValue value) {
    if (IsGpuMemory(value.GetTensorMemoryInfo())) {
      data_ = value.GetTensorRawData();
      return nullptr;
    }

    bytes_ = value.GetTensorSizeInBytes();
    if (bytes_ == 0) {
      data_ = value.GetTensorRawData();
      return nullptr;
    }

    musaError_t alloc_status = musaMalloc(&ptr_, bytes_);
    if (alloc_status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        MusaErrorString(alloc_status));
    }
    musaError_t status = musaMemcpy(ptr_, value.GetTensorRawData(), bytes_,
                                    musaMemcpyHostToDevice);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
    data_ = ptr_;
    return nullptr;
  }

  const void* data() const { return data_; }

 private:
  void* ptr_ = nullptr;
  const void* data_ = nullptr;
  size_t bytes_ = 0;
};

inline OrtStatus* CopyToHost(Ort::ConstValue value,
                             std::vector<uint8_t>& bytes) {
  size_t num_bytes = value.GetTensorSizeInBytes();
  bytes.resize(num_bytes);
  if (num_bytes == 0) {
    return nullptr;
  }

  const void* src = value.GetTensorRawData();
  if (IsGpuMemory(value.GetTensorMemoryInfo())) {
    musaError_t status =
        musaMemcpy(bytes.data(), src, num_bytes, musaMemcpyDeviceToHost);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
  } else {
    std::memcpy(bytes.data(), src, num_bytes);
  }

  return nullptr;
}

inline OrtStatus* CopyFromHost(Ort::UnownedValue value, const void* src,
                               size_t num_bytes) {
  if (num_bytes == 0) {
    return nullptr;
  }

  void* dst = value.GetTensorMutableRawData();
  if (IsGpuMemory(value.GetTensorMemoryInfo())) {
    musaError_t status =
        musaMemcpy(dst, src, num_bytes, musaMemcpyHostToDevice);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
  } else {
    std::memcpy(dst, src, num_bytes);
  }

  return nullptr;
}

inline OrtStatus* CopyRawTensor(Ort::ConstValue src_value,
                                Ort::UnownedValue dst_value, size_t num_bytes) {
  if (num_bytes == 0) {
    return nullptr;
  }
  const void* src = src_value.GetTensorRawData();
  void* dst = dst_value.GetTensorMutableRawData();
  if (src == dst) {
    return nullptr;
  }

  const bool src_gpu = IsGpuMemory(src_value.GetTensorMemoryInfo());
  const bool dst_gpu = IsGpuMemory(dst_value.GetTensorMemoryInfo());
  if (src_gpu && dst_gpu) {
    musaError_t status =
        musaMemcpyAsync(dst, src, num_bytes, musaMemcpyDeviceToDevice, nullptr);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
  } else if (src_gpu) {
    musaError_t status =
        musaMemcpy(dst, src, num_bytes, musaMemcpyDeviceToHost);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
  } else if (dst_gpu) {
    musaError_t status =
        musaMemcpy(dst, src, num_bytes, musaMemcpyHostToDevice);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
  } else {
    std::memcpy(dst, src, num_bytes);
  }
  return nullptr;
}

inline OrtStatus* DeviceMemcpy(void* dst, const void* src, size_t num_bytes) {
  if (num_bytes == 0 || dst == src) {
    return nullptr;
  }
  musaError_t status =
      musaMemcpyAsync(dst, src, num_bytes, musaMemcpyDeviceToDevice, nullptr);
  if (status != musaSuccess) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
  }
  return nullptr;
}

inline OrtStatus* DeviceMemcpy2D(void* dst, size_t dst_pitch, const void* src,
                                 size_t src_pitch, size_t width_bytes,
                                 size_t height) {
  if (width_bytes == 0 || height == 0) {
    return nullptr;
  }
  musaError_t status =
      musaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width_bytes, height,
                        musaMemcpyDeviceToDevice, nullptr);
  if (status != musaSuccess) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
  }
  return nullptr;
}

inline OrtStatus* LaunchStatus(musaError_t status) {
  if (status != musaSuccess) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
  }
  return nullptr;
}

inline bool ToMusaElementType(ONNXTensorElementDataType elem_type,
                              MusaElementType& musa_elem_type) {
  switch (elem_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      musa_elem_type = MusaElementType::Float;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      musa_elem_type = MusaElementType::Uint8;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      musa_elem_type = MusaElementType::Int8;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      musa_elem_type = MusaElementType::Uint16;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      musa_elem_type = MusaElementType::Int16;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      musa_elem_type = MusaElementType::Int32;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      musa_elem_type = MusaElementType::Int64;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      musa_elem_type = MusaElementType::Bool;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      musa_elem_type = MusaElementType::Float16;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      musa_elem_type = MusaElementType::Double;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      musa_elem_type = MusaElementType::Uint32;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      musa_elem_type = MusaElementType::Uint64;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      musa_elem_type = MusaElementType::BFloat16;
      return true;
    default:
      return false;
  }
}

inline OrtStatus* UnsupportedDeviceElementwiseStatus(
    const char* op_name, ONNXTensorElementDataType elem_type) {
  std::string message = std::string(op_name) +
                        " unsupported dtype or shape for MUSA device path: " +
                        std::to_string(static_cast<int>(elem_type));
  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, message.c_str());
}

inline bool AllGpuInputs(Ort::KernelContext& ctx) {
  for (size_t i = 0; i < ctx.GetInputCount(); ++i) {
    if (!IsGpuMemory(ctx.GetInput(i).GetTensorMemoryInfo())) {
      return false;
    }
  }
  return true;
}

inline size_t ElementSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return 2;
    default:
      return 0;
  }
}

// ---------------------------------------------------------------------------
// Shape arithmetic helpers.
// ---------------------------------------------------------------------------
inline int64_t NumElements(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 1;
  }
  int64_t n = 1;
  for (int64_t dim : shape) {
    n *= dim;
  }
  return n;
}

inline int64_t NormalizeAxis(int64_t axis, size_t rank) {
  int64_t r = static_cast<int64_t>(rank);
  return axis < 0 ? axis + r : axis;
}

inline std::vector<int64_t> Strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

inline std::vector<int64_t> Coordinates(int64_t linear,
                                        const std::vector<int64_t>& shape) {
  std::vector<int64_t> coord(shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    int64_t dim = shape[static_cast<size_t>(i)];
    coord[static_cast<size_t>(i)] = dim == 0 ? 0 : linear % dim;
    linear = dim == 0 ? 0 : linear / dim;
  }
  return coord;
}

inline int64_t Offset(const std::vector<int64_t>& coord,
                      const std::vector<int64_t>& strides) {
  int64_t off = 0;
  for (size_t i = 0; i < coord.size(); ++i) {
    off += coord[i] * strides[i];
  }
  return off;
}

inline std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& a,
                                           const std::vector<int64_t>& b) {
  size_t rank = std::max(a.size(), b.size());
  std::vector<int64_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (da != db && da != 1 && db != 1) {
      throw std::runtime_error("broadcast shape mismatch");
    }
    out[i] = da == 0 || db == 0 ? 0 : std::max(da, db);
  }
  return out;
}

inline MusaBroadcastParams MakeBroadcastParams(
    const std::vector<int64_t>& out_shape,
    const std::vector<int64_t>& lhs_shape,
    const std::vector<int64_t>& rhs_shape) {
  MusaBroadcastParams params{};
  const size_t rank = out_shape.size();
  params.rank = static_cast<int32_t>(rank);
  params.total_elements = NumElements(out_shape);

  auto out_strides = Strides(out_shape);
  auto lhs_strides = Strides(lhs_shape);
  auto rhs_strides = Strides(rhs_shape);
  const size_t lhs_rank = lhs_shape.size();
  const size_t rhs_rank = rhs_shape.size();
  const size_t lhs_offset = rank - lhs_rank;
  const size_t rhs_offset = rank - rhs_rank;

  for (size_t dim = 0; dim < rank; ++dim) {
    params.output_strides[dim] = out_strides[dim];

    if (dim < lhs_offset) {
      params.lhs_strides[dim] = 0;
    } else {
      const size_t lhs_dim = dim - lhs_offset;
      params.lhs_strides[dim] =
          lhs_shape[lhs_dim] == 1 ? 0 : lhs_strides[lhs_dim];
    }

    if (dim < rhs_offset) {
      params.rhs_strides[dim] = 0;
    } else {
      const size_t rhs_dim = dim - rhs_offset;
      params.rhs_strides[dim] =
          rhs_shape[rhs_dim] == 1 ? 0 : rhs_strides[rhs_dim];
    }
  }

  return params;
}

inline bool CanUseBroadcastKernel(const std::vector<int64_t>& out_shape,
                                  const std::vector<int64_t>& lhs_shape,
                                  const std::vector<int64_t>& rhs_shape) {
  return out_shape.size() <= kMusaMaxBroadcastRank &&
         lhs_shape.size() <= kMusaMaxBroadcastRank &&
         rhs_shape.size() <= kMusaMaxBroadcastRank;
}

inline int64_t BroadcastOffset(const std::vector<int64_t>& out_coord,
                               const std::vector<int64_t>& in_shape,
                               const std::vector<int64_t>& in_strides) {
  size_t rank = out_coord.size();
  size_t in_rank = in_shape.size();
  int64_t off = 0;
  for (size_t i = 0; i < in_rank; ++i) {
    size_t out_i = rank - in_rank + i;
    int64_t c = in_shape[i] == 1 ? 0 : out_coord[out_i];
    off += c * in_strides[i];
  }
  return off;
}

inline std::vector<int64_t> PrefixShape(const std::vector<int64_t>& shape,
                                        size_t trailing_dims) {
  if (shape.size() < trailing_dims) {
    return {};
  }
  return std::vector<int64_t>(
      shape.begin(), shape.end() - static_cast<int64_t>(trailing_dims));
}

inline std::vector<int64_t> BroadcastBatchCoord(
    const std::vector<int64_t>& out_coord,
    const std::vector<int64_t>& out_shape,
    const std::vector<int64_t>& in_shape) {
  std::vector<int64_t> coord(in_shape.size(), 0);
  size_t out_rank = out_shape.size();
  size_t in_rank = in_shape.size();
  for (size_t i = 0; i < in_rank; ++i) {
    size_t out_i = out_rank - in_rank + i;
    coord[i] = in_shape[i] == 1 ? 0 : out_coord[out_i];
  }
  return coord;
}

inline std::set<int64_t> AxesSet(std::vector<int64_t> axes, size_t rank) {
  std::set<int64_t> out;
  if (axes.empty()) {
    for (size_t i = 0; i < rank; ++i) out.insert(static_cast<int64_t>(i));
    return out;
  }
  for (int64_t axis : axes) {
    out.insert(NormalizeAxis(axis, rank));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Typed tensor read/write helpers.
// ---------------------------------------------------------------------------
template <typename T>
std::span<const T> Span(const std::vector<uint8_t>& bytes) {
  return std::span<const T>(reinterpret_cast<const T*>(bytes.data()),
                            bytes.size() / sizeof(T));
}

template <typename T>
std::vector<T> ReadTyped(Ort::ConstValue value) {
  std::vector<uint8_t> bytes;
  Ort::ThrowOnError(CopyToHost(value, bytes));
  std::vector<T> out(bytes.size() / sizeof(T));
  if (!out.empty()) {
    std::memcpy(out.data(), bytes.data(), bytes.size());
  }
  return out;
}

template <typename T>
OrtStatus* WriteTyped(Ort::UnownedValue value, const std::vector<T>& data) {
  return CopyFromHost(value, data.data(), data.size() * sizeof(T));
}

inline std::vector<int64_t> ReadIntTensor(Ort::KernelContext& ctx,
                                          size_t index) {
  Ort::ConstValue value = ctx.GetInput(index);
  auto info = value.GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return ReadTyped<int64_t>(value);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> vals = ReadTyped<int32_t>(value);
    return std::vector<int64_t>(vals.begin(), vals.end());
  }
  throw std::runtime_error("expected int32/int64 tensor");
}

// ---------------------------------------------------------------------------
// Element-wise compute helpers shared by the binary / unary kernels.
// ---------------------------------------------------------------------------
template <typename T, typename Fn>
OrtStatus* BinaryCompute(Ort::KernelContext& ctx,
                         const std::vector<int64_t>& shape0,
                         const std::vector<int64_t>& shape1, Fn fn) {
  std::vector<T> a = ReadTyped<T>(ctx.GetInput(0));
  std::vector<T> b = ReadTyped<T>(ctx.GetInput(1));
  std::vector<int64_t> out_shape = BroadcastShape(shape0, shape1);
  int64_t total = NumElements(out_shape);
  std::vector<T> out(static_cast<size_t>(total));

  auto s0 = Strides(shape0);
  auto s1 = Strides(shape1);
  for (int64_t i = 0; i < total; ++i) {
    auto coord = Coordinates(i, out_shape);
    int64_t o0 = BroadcastOffset(coord, shape0, s0);
    int64_t o1 = BroadcastOffset(coord, shape1, s1);
    out[static_cast<size_t>(i)] =
        fn(a[static_cast<size_t>(o0)], b[static_cast<size_t>(o1)]);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return WriteTyped<T>(y, out);
}

template <typename T, typename Fn>
OrtStatus* UnaryCompute(Ort::KernelContext& ctx,
                        const std::vector<int64_t>& shape, Fn fn) {
  std::vector<T> x = ReadTyped<T>(ctx.GetInput(0));
  std::vector<T> y_data(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    y_data[i] = fn(x[i]);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  return WriteTyped<T>(y, y_data);
}

template <typename T, typename Fn>
OrtStatus* BinaryCompute(Ort::KernelContext& ctx,
                         const std::vector<int64_t>& shape0,
                         const std::vector<int64_t>& shape1, Fn fn,
                         MusaBinaryOp device_op) {
  if constexpr (std::is_same_v<T, float>) {
    std::vector<int64_t> out_shape = BroadcastShape(shape0, shape1);
    Ort::ConstValue lhs = ctx.GetInput(0);
    Ort::ConstValue rhs = ctx.GetInput(1);
    if (IsGpuMemory(lhs.GetTensorMemoryInfo()) &&
        IsGpuMemory(rhs.GetTensorMemoryInfo()) &&
        CanUseBroadcastKernel(out_shape, shape0, shape1)) {
      Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
      if (IsGpuMemory(y.GetTensorMemoryInfo())) {
        MusaBroadcastParams params =
            MakeBroadcastParams(out_shape, shape0, shape1);
        return LaunchStatus(LaunchMusaBinaryFloatKernel(
            lhs.GetTensorData<float>(), rhs.GetTensorData<float>(),
            y.GetTensorMutableData<float>(), params, device_op, nullptr));
      }
    }
  }
  return BinaryCompute<T>(ctx, shape0, shape1, fn);
}

template <typename T, typename Fn>
OrtStatus* UnaryCompute(Ort::KernelContext& ctx,
                        const std::vector<int64_t>& shape, Fn fn,
                        MusaUnaryOp device_op, float alpha = 0.0f) {
  if constexpr (std::is_same_v<T, float>) {
    Ort::ConstValue input = ctx.GetInput(0);
    if (IsGpuMemory(input.GetTensorMemoryInfo())) {
      Ort::UnownedValue y = ctx.GetOutput(0, shape);
      if (IsGpuMemory(y.GetTensorMemoryInfo())) {
        return LaunchStatus(LaunchMusaUnaryFloatKernel(
            input.GetTensorData<float>(), y.GetTensorMutableData<float>(),
            NumElements(shape), device_op, alpha, nullptr));
      }
    }
  }
  return UnaryCompute<T>(ctx, shape, fn);
}

inline OrtStatus* BinaryDeviceCompute(Ort::KernelContext& ctx,
                                      const std::vector<int64_t>& shape0,
                                      const std::vector<int64_t>& shape1,
                                      ONNXTensorElementDataType elem_type,
                                      MusaBinaryOp device_op,
                                      const char* op_name) {
  std::vector<int64_t> out_shape = BroadcastShape(shape0, shape1);
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      !IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo()) ||
      !CanUseBroadcastKernel(out_shape, shape0, shape1)) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }

  musaError_t status =
      LaunchMusaBinaryKernel(lhs.GetTensorRawData(), rhs.GetTensorRawData(),
                             y.GetTensorMutableRawData(),
                             MakeBroadcastParams(out_shape, shape0, shape1),
                             device_op, musa_elem_type, nullptr);
  if (status == musaErrorNotSupported) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }
  return LaunchStatus(status);
}

inline OrtStatus* UnaryDeviceCompute(Ort::KernelContext& ctx,
                                     const std::vector<int64_t>& shape,
                                     ONNXTensorElementDataType elem_type,
                                     MusaUnaryOp device_op, const char* op_name,
                                     float alpha = 0.0f) {
  Ort::ConstValue input = ctx.GetInput(0);
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      !IsGpuMemory(input.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }

  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }

  musaError_t status = LaunchMusaUnaryKernel(
      input.GetTensorRawData(), y.GetTensorMutableRawData(), NumElements(shape),
      device_op, alpha, musa_elem_type, nullptr);
  if (status == musaErrorNotSupported) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }
  return LaunchStatus(status);
}

inline OrtStatus* CompareDeviceCompute(Ort::KernelContext& ctx,
                                       const std::vector<int64_t>& shape0,
                                       const std::vector<int64_t>& shape1,
                                       ONNXTensorElementDataType elem_type,
                                       MusaCompareOp device_op,
                                       const char* op_name) {
  std::vector<int64_t> out_shape = BroadcastShape(shape0, shape1);
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      !IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo()) ||
      !CanUseBroadcastKernel(out_shape, shape0, shape1)) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }

  musaError_t status =
      LaunchMusaCompareKernel(lhs.GetTensorRawData(), rhs.GetTensorRawData(),
                              y.GetTensorMutableData<uint8_t>(),
                              MakeBroadcastParams(out_shape, shape0, shape1),
                              device_op, musa_elem_type, nullptr);
  if (status == musaErrorNotSupported) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }
  return LaunchStatus(status);
}
