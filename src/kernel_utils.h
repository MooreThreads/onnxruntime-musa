#pragma once

#include <numeric>
#include <vector>

#include "plugin_ep_utils.h"

inline const OrtDataType* TensorDataType(
    ONNXTensorElementDataType element_type) {
  const OrtDataType* result = nullptr;
  Ort::ThrowOnError(Ort::GetEpApi().GetTensorDataType(element_type, &result));
  return result;
}

struct KernelCreateInfo {
  Ort::KernelDef kernel_def{nullptr};
  OrtKernelCreateFunc kernel_create_func = nullptr;
  void* kernel_create_func_state = nullptr;
};

using BuildKernelCreateInfoFn = OrtStatus* (*)(const char*, void*,
                                               KernelCreateInfo*);

template <typename T>
OrtStatus* BuildKernelCreateInfo(const char* ep_name, void* create_func_state,
                                 KernelCreateInfo* result);

template <typename T>
OrtStatus* GetInputDataAndShape(Ort::KernelContext& context, size_t index,
                                const T*& data, std::vector<int64_t>& shape,
                                size_t& element_count) {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstValue input = context.GetInput(index);
  auto type_shape = input.GetTensorTypeAndShapeInfo();
  shape = type_shape.GetShape();
  element_count = type_shape.GetElementCount();
  data = input.GetTensorData<T>();
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

inline bool HasStaticShape(const std::vector<int64_t>& shape) {
  for (int64_t dim : shape) {
    if (dim < 0) {
      return false;
    }
  }
  return true;
}

inline int64_t ShapeSize(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>{});
}

inline bool BroadcastShapes(const std::vector<int64_t>& lhs,
                            const std::vector<int64_t>& rhs,
                            std::vector<int64_t>& output) {
  if (!HasStaticShape(lhs) || !HasStaticShape(rhs)) {
    return false;
  }

  const size_t rank = std::max(lhs.size(), rhs.size());
  output.assign(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t lhs_dim =
        i < rank - lhs.size() ? 1 : lhs[i - (rank - lhs.size())];
    const int64_t rhs_dim =
        i < rank - rhs.size() ? 1 : rhs[i - (rank - rhs.size())];
    if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
      return false;
    }
    output[i] = std::max(lhs_dim, rhs_dim);
  }
  return true;
}

inline std::vector<int64_t> ContiguousStrides(
    const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}
