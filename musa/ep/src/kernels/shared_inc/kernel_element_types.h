// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <string>

#include "shared_inc/device_kernel_types.h"
#include "utils.h"

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
