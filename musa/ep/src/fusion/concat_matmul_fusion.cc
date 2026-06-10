// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/concat_matmul_fusion.h"

#include <mudnn.h>
#include <musa_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "kernels/math/matmul.h"
#include "kernels/shared_inc/blas_utils.h"

/*
 * ConcatMatMul Fusion Pattern
 *
 * Matches one of the following ONNX subgraphs after ORT graph partitioning:
 *
 *   Y = MatMul(Concat(X0, X1, ..., axis), B)
 *
 * or:
 *
 *   Y = MatMul(A, Concat(X0, X1, ..., axis))
 *
 * Constraints:
 *   - all tensors are float32
 *   - Concat has at least two inputs
 *   - MatMul inputs have the same rank and rank >= 2
 *   - batch dimensions match
 *   - Concat output is consumed only by the matched MatMul
 *
 * Semantics:
 *   output = MatMul(concat_inputs_along_axis, other_input)
 *   or
 *   output = MatMul(other_input, concat_inputs_along_axis)
 *
 * Runtime path:
 *   - run Concat on MUSA with muDNN
 *   - feed the concatenated temporary directly into the existing MUSA MatMul
 *     path, including the 4D-to-3D/batched muDNN fast path for model hot spots
 */

class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t bytes, musaStream_t stream = nullptr) {
    Resize(bytes, stream);
  }

  void Resize(size_t bytes, musaStream_t stream) {
    if (bytes <= bytes_) {
      stream_ = stream;
      return;
    }

    if (ptr_ != nullptr) {
      FreeDeviceMemoryOnStream(ptr_, stream_);
      ptr_ = nullptr;
      bytes_ = 0;
    }

    stream_ = stream;
    if (bytes == 0) {
      return;
    }

    musaError_t status = musaMalloc(&ptr_, bytes);
    if (status != musaSuccess) {
      throw std::runtime_error(MusaErrorString(status));
    }
    bytes_ = bytes;
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      // This scratch buffer can outlive the run compute stream. Do not defer
      // teardown frees through stream events; the saved stream may be stale.
      (void)musaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get() const { return ptr_; }
  size_t bytes() const { return bytes_; }

  template <typename T>
  T* data() const {
    return reinterpret_cast<T*>(ptr_);
  }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
  musaStream_t stream_ = nullptr;
};

struct ConcatMatMulScratch {
  DeviceBuffer concat_buffer;
  DeviceBuffer other_buffer;
  DeviceBuffer output_buffer;
  std::vector<std::unique_ptr<DeviceBuffer>> workspace_buffers;
  size_t workspace_index = 0;

  void ResetWorkspace() { workspace_index = 0; }

  void* Workspace(size_t bytes, musaStream_t stream) {
    if (bytes == 0) {
      return nullptr;
    }

    if (workspace_index == workspace_buffers.size()) {
      workspace_buffers.push_back(std::make_unique<DeviceBuffer>());
    }

    DeviceBuffer& buffer = *workspace_buffers[workspace_index++];
    buffer.Resize(bytes, stream);
    return buffer.get();
  }
};

namespace {

void NoOpDelete(void*) {}

::musa::dnn::Handle* MudnnHandleOrThrow(musaStream_t stream) {
  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* status = EnsureMudnnHandle(&handle, stream);
  if (status != nullptr) {
    Ort::GetApi().ReleaseStatus(status);
    throw std::runtime_error("mudnn Handle SetAllowTF32 failed");
  }
  return handle;
}

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(const Ort::ConstNode& node, const char* op_type) {
  return node.GetOperatorType() == op_type && IsOnnxDomain(node.GetDomain());
}

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
}

int64_t ReadIntAttribute(Ort::ConstNode node, const std::string& name) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK()) {
    throw std::runtime_error("missing required node attribute: " + name);
  }

  int64_t value = 0;
  status = attr.GetValue(value);
  if (!status.IsOK()) {
    throw std::runtime_error("failed to read node attribute: " + name);
  }

  return value;
}

int64_t NormalizeAxisChecked(int64_t axis, size_t rank) {
  const int64_t signed_rank = static_cast<int64_t>(rank);
  if (rank == 0 || axis < -signed_rank || axis >= signed_rank) {
    throw std::runtime_error("ConcatMatMul axis is out of range");
  }
  return axis < 0 ? axis + signed_rank : axis;
}

std::vector<int64_t> TensorShape(Ort::ConstValue value) {
  return value.GetTensorTypeAndShapeInfo().GetShape();
}

void ValidateFloatTensor(Ort::ConstValue value, const char* name) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(
        std::string("ConcatMatMul only supports float tensors for ") + name);
  }
}

std::vector<int64_t> ComputeConcatShape(
    const std::vector<Ort::ConstValue>& concat_inputs, int64_t axis) {
  if (concat_inputs.size() < 2) {
    throw std::runtime_error(
        "ConcatMatMul requires at least two concat inputs");
  }

  std::vector<int64_t> concat_shape = TensorShape(concat_inputs[0]);
  if (concat_shape.size() < 2) {
    throw std::runtime_error("ConcatMatMul requires concat input rank >= 2");
  }

  axis = NormalizeAxisChecked(axis, concat_shape.size());
  concat_shape[static_cast<size_t>(axis)] = 0;
  for (Ort::ConstValue input : concat_inputs) {
    ValidateFloatTensor(input, "concat input");
    std::vector<int64_t> shape = TensorShape(input);
    if (shape.size() != concat_shape.size()) {
      throw std::runtime_error("ConcatMatMul concat input rank mismatch");
    }
    for (size_t dim = 0; dim < shape.size(); ++dim) {
      if (dim == static_cast<size_t>(axis)) {
        continue;
      }
      if (shape[dim] != concat_shape[dim]) {
        throw std::runtime_error("ConcatMatMul concat non-axis shape mismatch");
      }
    }
    concat_shape[static_cast<size_t>(axis)] += shape[static_cast<size_t>(axis)];
  }
  return concat_shape;
}

std::vector<int64_t> ComputeMatMulOutputShape(
    const std::vector<int64_t>& lhs_shape,
    const std::vector<int64_t>& rhs_shape) {
  if (lhs_shape.size() < 2 || rhs_shape.size() < 2 ||
      lhs_shape.size() != rhs_shape.size()) {
    throw std::runtime_error(
        "ConcatMatMul currently requires MatMul inputs with equal rank >= 2");
  }

  const size_t rank = lhs_shape.size();
  for (size_t dim = 0; dim + 2 < rank; ++dim) {
    if (lhs_shape[dim] != rhs_shape[dim]) {
      throw std::runtime_error("ConcatMatMul batch dimensions must match");
    }
  }
  if (lhs_shape[rank - 1] != rhs_shape[rank - 2]) {
    throw std::runtime_error("ConcatMatMul MatMul K dimension mismatch");
  }

  std::vector<int64_t> output_shape = lhs_shape;
  output_shape[rank - 2] = lhs_shape[rank - 2];
  output_shape[rank - 1] = rhs_shape[rank - 1];
  return output_shape;
}

bool IsGpuValue(Ort::ConstValue value) {
  return IsGpuMemory(value.GetTensorMemoryInfo());
}

std::vector<int64_t> ShapeStrides(const std::vector<int64_t>& shape);

bool AllGpuValues(const std::vector<Ort::ConstValue>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](Ort::ConstValue value) { return IsGpuValue(value); });
}

std::vector<int64_t> ShapeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  if (shape.empty()) {
    return strides;
  }
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

::musa::dnn::Tensor::Format MudnnFormatForShape(
    const std::vector<int64_t>& shape) {
  if (shape.empty() || shape.size() == 1 || shape.size() == 3) {
    return ::musa::dnn::Tensor::Format::NCW;
  }
  if (shape.size() == 4) {
    return ::musa::dnn::Tensor::Format::NCHW;
  }
  if (shape.size() == 5) {
    return ::musa::dnn::Tensor::Format::NCDHW;
  }
  return ::musa::dnn::Tensor::Format::NCHW;
}

void CheckMudnnStatus(::musa::dnn::Status status, const char* message) {
  if (status != ::musa::dnn::Status::SUCCESS) {
    throw std::runtime_error(std::string(message) + ", status: " +
                             std::to_string(static_cast<int>(status)));
  }
}

void SetupMudnnTensor(::musa::dnn::Tensor& tensor, const void* data,
                      const std::vector<int64_t>& shape) {
  CheckMudnnStatus(tensor.SetType(::musa::dnn::Tensor::Type::FLOAT),
                   "Failed to set ConcatMatMul mudnn tensor type");
  if (data != nullptr) {
    CheckMudnnStatus(tensor.SetAddr(data),
                     "Failed to set ConcatMatMul mudnn tensor address");
  } else if (NumElements(shape) > 0) {
    throw std::runtime_error(
        "ConcatMatMul mudnn tensor data pointer is null for non-empty tensor");
  }

  CheckMudnnStatus(tensor.SetFormat(MudnnFormatForShape(shape)),
                   "Failed to set ConcatMatMul mudnn tensor format");

  std::vector<int64_t> dims = shape.empty() ? std::vector<int64_t>{1} : shape;
  std::vector<int64_t> strides =
      shape.empty() ? std::vector<int64_t>{1} : ShapeStrides(shape);
  CheckMudnnStatus(tensor.SetNdInfo(static_cast<int>(dims.size()), dims.data(),
                                    strides.data()),
                   "Failed to set ConcatMatMul mudnn tensor shape");
}

void SetupMudnnTensorCompact(::musa::dnn::Tensor& tensor, const void* data,
                             const std::vector<int64_t>& shape,
                             ::musa::dnn::Tensor::Format format) {
  CheckMudnnStatus(tensor.SetType(::musa::dnn::Tensor::Type::FLOAT),
                   "Failed to set ConcatMatMul mudnn tensor type");
  if (data != nullptr) {
    CheckMudnnStatus(tensor.SetAddr(data),
                     "Failed to set ConcatMatMul mudnn tensor address");
  } else if (NumElements(shape) > 0) {
    throw std::runtime_error(
        "ConcatMatMul mudnn tensor data pointer is null for non-empty tensor");
  }

  CheckMudnnStatus(tensor.SetFormat(format),
                   "Failed to set ConcatMatMul mudnn tensor format");

  std::vector<int64_t> dims = shape.empty() ? std::vector<int64_t>{1} : shape;
  CheckMudnnStatus(tensor.SetNdInfo(static_cast<int>(dims.size()), dims.data()),
                   "Failed to set ConcatMatMul mudnn tensor shape");
}

bool CanReshape4DTo3D(const std::vector<int64_t>& lhs_shape,
                      const std::vector<int64_t>& rhs_shape) {
  if (lhs_shape.size() != 4 || rhs_shape.size() != 4) {
    return false;
  }
  if (lhs_shape[0] != rhs_shape[0] || lhs_shape[1] != rhs_shape[1]) {
    return false;
  }
  for (int64_t dim : lhs_shape) {
    if (dim <= 0) {
      return false;
    }
  }
  for (int64_t dim : rhs_shape) {
    if (dim <= 0) {
      return false;
    }
  }
  return lhs_shape[0] <= INT64_MAX / lhs_shape[1];
}

std::vector<int64_t> Reshape4DTo3D(const std::vector<int64_t>& shape) {
  return {shape[0] * shape[1], shape[2], shape[3]};
}

OrtStatus* RunMudnnConcatMatMulBatched(const float* a_data, const float* b_data,
                                       float* y_data,
                                       const std::vector<int64_t>& lhs_shape,
                                       const std::vector<int64_t>& rhs_shape,
                                       const std::vector<int64_t>& output_shape,
                                       ConcatMatMulScratch& scratch,
                                       musaStream_t stream) {
  if (lhs_shape.size() <= 2 && rhs_shape.size() <= 2) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "ConcatMatMul batched path requires rank > 2 inputs");
  }

  const bool reshape_4d_to_3d = CanReshape4DTo3D(lhs_shape, rhs_shape);
  std::vector<int64_t> lhs_batch_shape =
      reshape_4d_to_3d ? Reshape4DTo3D(lhs_shape) : lhs_shape;
  std::vector<int64_t> rhs_batch_shape =
      reshape_4d_to_3d ? Reshape4DTo3D(rhs_shape) : rhs_shape;
  std::vector<int64_t> y_batch_shape =
      reshape_4d_to_3d ? Reshape4DTo3D(output_shape) : output_shape;

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, stream);
  if (handle_status != nullptr) {
    return handle_status;
  }

  ::musa::dnn::Tensor lhs_tensor;
  ::musa::dnn::Tensor rhs_tensor;
  ::musa::dnn::Tensor y_tensor;
  if (reshape_4d_to_3d) {
    SetupMudnnTensorCompact(lhs_tensor, a_data, lhs_batch_shape,
                            ::musa::dnn::Tensor::Format::NCHW);
    SetupMudnnTensorCompact(rhs_tensor, b_data, rhs_batch_shape,
                            ::musa::dnn::Tensor::Format::NCHW);
    SetupMudnnTensorCompact(y_tensor, y_data, y_batch_shape,
                            ::musa::dnn::Tensor::Format::NCHW);
  } else {
    SetupMudnnTensor(lhs_tensor, a_data, lhs_batch_shape);
    SetupMudnnTensor(rhs_tensor, b_data, rhs_batch_shape);
    SetupMudnnTensor(y_tensor, y_data, y_batch_shape);
  }

  scratch.ResetWorkspace();
  auto memory_allocator =
      [&scratch, stream](size_t size) -> ::musa::dnn::MemoryHandler {
    if (size == 0) {
      return ::musa::dnn::MemoryHandler(nullptr, NoOpDelete);
    }
    return ::musa::dnn::MemoryHandler(scratch.Workspace(size, stream),
                                      NoOpDelete);
  };

  ::musa::dnn::BatchMatMul batch_op;
  CheckMudnnStatus(batch_op.SetAlpha(1.0),
                   "Failed to set ConcatMatMul BatchMatMul alpha");
  CheckMudnnStatus(batch_op.SetBeta(0.0),
                   "Failed to set ConcatMatMul BatchMatMul beta");
  CheckMudnnStatus(batch_op.SetTranspose(false, false),
                   "Failed to set ConcatMatMul BatchMatMul transpose");
  CheckMudnnStatus(
      batch_op.SetComputeMode(::musa::dnn::BatchMatMul::ComputeMode::TENSOR),
      "Failed to set ConcatMatMul BatchMatMul compute mode");
  CheckMudnnStatus(
      batch_op.Run(*handle, y_tensor, lhs_tensor, rhs_tensor, memory_allocator),
      "ConcatMatMul BatchMatMul execution failed");
  return nullptr;
}

void RunMudnnConcat(const std::vector<Ort::ConstValue>& concat_inputs,
                    const std::vector<int64_t>& concat_shape, int64_t axis,
                    float* concat_data, musaStream_t stream) {
  ::musa::dnn::Handle* handle = MudnnHandleOrThrow(stream);
  std::vector<::musa::dnn::Tensor> input_tensors(concat_inputs.size());
  for (size_t i = 0; i < concat_inputs.size(); ++i) {
    SetupMudnnTensor(input_tensors[i], concat_inputs[i].GetTensorData<float>(),
                     TensorShape(concat_inputs[i]));
  }

  ::musa::dnn::Tensor output_tensor;
  SetupMudnnTensor(output_tensor, concat_data, concat_shape);

  ::musa::dnn::Concat concat_op;
  CheckMudnnStatus(concat_op.SetAxis(static_cast<int>(axis)),
                   "Failed to set ConcatMatMul mudnn concat axis");
  CheckMudnnStatus(concat_op.Run(*handle, output_tensor,
                                 static_cast<int>(input_tensors.size()),
                                 input_tensors.data()),
                   "ConcatMatMul mudnn concat execution failed");
}

OrtStatus* ComputeDeviceConcatMatMul(
    Ort::UnownedValue y, const std::vector<Ort::ConstValue>& concat_inputs,
    Ort::ConstValue other_input, const std::vector<int64_t>& concat_shape,
    const std::vector<int64_t>& other_shape, int64_t axis,
    int64_t concat_input_idx, const std::vector<int64_t>& output_shape,
    ConcatMatMulScratch& scratch, musaStream_t stream) {
  scratch.concat_buffer.Resize(static_cast<size_t>(NumElements(concat_shape)) *
                                   sizeof(float),
                               stream);
  DeviceBuffer& concat_buffer = scratch.concat_buffer;
  if (!AllGpuValues(concat_inputs)) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "ConcatMatMul requires MUSA concat inputs");
  }
  RunMudnnConcat(concat_inputs, concat_shape, axis, concat_buffer.data<float>(),
                 stream);

  if (!IsGpuValue(other_input)) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "ConcatMatMul requires MUSA MatMul input");
  }
  const float* other_data = other_input.GetTensorData<float>();

  const std::vector<int64_t>& lhs_shape =
      concat_input_idx == 0 ? concat_shape : other_shape;
  const std::vector<int64_t>& rhs_shape =
      concat_input_idx == 0 ? other_shape : concat_shape;
  const float* a_data =
      concat_input_idx == 0 ? concat_buffer.data<float>() : other_data;
  const float* b_data =
      concat_input_idx == 0 ? other_data : concat_buffer.data<float>();
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "ConcatMatMul requires MUSA output");
  }
  float* y_data = y.GetTensorMutableData<float>();

  const size_t rank = lhs_shape.size();
  if (lhs_shape[rank - 2] > INT32_MAX || lhs_shape[rank - 1] > INT32_MAX ||
      rhs_shape[rank - 1] > INT32_MAX) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "ConcatMatMul dimensions exceed int32 limits");
  }

  const bool used_mudnn_batched = lhs_shape.size() > 2 || rhs_shape.size() > 2;
  OrtStatus* matmul_status =
      used_mudnn_batched
          ? RunMudnnConcatMatMulBatched(a_data, b_data, y_data, lhs_shape,
                                        rhs_shape, output_shape, scratch,
                                        stream)
          : ComputeMusaMatMulDevice(a_data, b_data, y_data, lhs_shape,
                                    rhs_shape, output_shape, stream);
  if (matmul_status != nullptr) {
    return matmul_status;
  }
  return nullptr;
}

}  // namespace

std::unique_ptr<FusionNodeCompute> CreateConcatMatMulFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  if (nodes.size() != 2) {
    throw std::runtime_error("ConcatMatMul fusion expects exactly two nodes");
  }

  Ort::ConstNode concat_node{nullptr};
  Ort::ConstNode matmul_node{nullptr};
  for (Ort::ConstNode node : nodes) {
    if (IsOnnxOp(node, "Concat")) {
      concat_node = node;
    } else if (IsOnnxOp(node, "MatMul")) {
      matmul_node = node;
    }
  }

  if (!concat_node || !matmul_node) {
    throw std::runtime_error("ConcatMatMul fusion expects Concat and MatMul");
  }

  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> matmul_inputs = matmul_node.GetInputs();
  if (concat_inputs.size() < 2 || concat_outputs.size() != 1 ||
      matmul_inputs.size() != 2) {
    throw std::runtime_error("invalid ConcatMatMul fused graph");
  }

  const std::string concat_output_name = Name(concat_outputs[0]);
  int64_t concat_input_idx = -1;
  if (Name(matmul_inputs[0]) == concat_output_name) {
    concat_input_idx = 0;
  } else if (Name(matmul_inputs[1]) == concat_output_name) {
    concat_input_idx = 1;
  } else {
    throw std::runtime_error("Concat output does not feed MatMul");
  }

  std::vector<Ort::ConstValueInfo> fused_inputs = fused_node.GetInputs();
  std::unordered_map<std::string, size_t> fused_input_indices;
  for (size_t i = 0; i < fused_inputs.size(); ++i) {
    fused_input_indices.emplace(Name(fused_inputs[i]), i);
  }

  std::vector<size_t> concat_input_indices;
  concat_input_indices.reserve(concat_inputs.size());
  for (Ort::ConstValueInfo input : concat_inputs) {
    auto it = fused_input_indices.find(Name(input));
    if (it == fused_input_indices.end()) {
      throw std::runtime_error(
          "unable to map Concat input to fused node input");
    }
    concat_input_indices.push_back(it->second);
  }

  auto other_input_name =
      Name(matmul_inputs[static_cast<size_t>(1 - concat_input_idx)]);
  auto other_it = fused_input_indices.find(other_input_name);
  if (other_it == fused_input_indices.end()) {
    throw std::runtime_error("unable to map MatMul input to fused node input");
  }

  return std::make_unique<ConcatMatMulFusionCompute>(
      ReadIntAttribute(concat_node, "axis"), concat_input_idx,
      std::move(concat_input_indices), other_it->second);
}

ConcatMatMulFusionCompute::ConcatMatMulFusionCompute(
    int64_t axis, int64_t concat_input_idx,
    std::vector<size_t> concat_input_indices, size_t other_input_index)
    : axis(axis),
      concat_input_idx(concat_input_idx),
      concat_input_indices(std::move(concat_input_indices)),
      other_input_index(other_input_index) {}

ConcatMatMulFusionCompute::~ConcatMatMulFusionCompute() = default;

OrtStatus* ConcatMatMulFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    musaStream_t stream = GetComputeStream(ctx);
    std::vector<Ort::ConstValue> concat_inputs;
    concat_inputs.reserve(concat_input_indices.size());
    for (size_t index : concat_input_indices) {
      concat_inputs.push_back(ctx.GetInput(index));
    }
    Ort::ConstValue other_input = ctx.GetInput(other_input_index);

    ValidateFloatTensor(other_input, "MatMul input");
    std::vector<int64_t> concat_shape = ComputeConcatShape(concat_inputs, axis);
    const int64_t normalized_axis =
        NormalizeAxisChecked(axis, concat_shape.size());
    std::vector<int64_t> other_shape = TensorShape(other_input);
    if (other_shape.size() != concat_shape.size()) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "ConcatMatMul currently requires MatMul inputs with equal rank");
    }

    const std::vector<int64_t>& lhs_shape =
        concat_input_idx == 0 ? concat_shape : other_shape;
    const std::vector<int64_t>& rhs_shape =
        concat_input_idx == 0 ? other_shape : concat_shape;
    std::vector<int64_t> output_shape =
        ComputeMatMulOutputShape(lhs_shape, rhs_shape);

    Ort::UnownedValue y = ctx.GetOutput(0, output_shape);
    std::lock_guard<std::mutex> lock(scratch_mutex);
    if (!scratch) {
      scratch = std::make_unique<ConcatMatMulScratch>();
    }
    return ComputeDeviceConcatMatMul(
        y, concat_inputs, other_input, concat_shape, other_shape,
        normalized_axis, concat_input_idx, output_shape, *scratch, stream);
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}
