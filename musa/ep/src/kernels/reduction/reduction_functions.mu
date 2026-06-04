#include "reduction/reduction_functions.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ int64_t ReduceInputBase(int64_t output_index,
                                                   MusaReduceParams params) {
  int64_t remaining = output_index;
  int64_t input_base = 0;
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    if (params.reduce_axes[dim] != 0) {
      continue;
    }
    const int64_t stride = params.output_strides[dim];
    const int64_t coord = stride == 0 ? 0 : remaining / stride;
    if (stride != 0) {
      remaining -= coord * stride;
    }
    input_base += coord * params.input_strides[dim];
  }
  return input_base;
}

__device__ __forceinline__ int64_t ReduceInputOffset(int64_t reduction_index,
                                                     MusaReduceParams params) {
  int64_t remaining = reduction_index;
  int64_t input_offset = 0;
  for (int32_t dim = params.rank - 1; dim >= 0; --dim) {
    if (params.reduce_axes[dim] == 0) {
      continue;
    }
    const int64_t coord = remaining % params.input_dims[dim];
    remaining /= params.input_dims[dim];
    input_offset += coord * params.input_strides[dim];
  }
  return input_offset;
}

__device__ __forceinline__ float ReduceInitValue(MusaReduceOp op) {
  return op == MusaReduceOp::Prod ? 1.0f : 0.0f;
}

__device__ __forceinline__ float ReduceUpdateValue(float acc, float value, MusaReduceOp op) {
  if (op == MusaReduceOp::Prod) {
    return acc * value;
  }
  if (op == MusaReduceOp::SumSquare) {
    return acc + value * value;
  }
  return acc + value;
}

__device__ __forceinline__ float ReduceCombineValue(float lhs, float rhs, MusaReduceOp op) {
  if (op == MusaReduceOp::Prod) {
    return lhs * rhs;
  }
  return lhs + rhs;
}

__global__ void ReduceFloatKernel(const float* input,
                                  float* output,
                                  MusaReduceParams params,
                                  MusaReduceOp op) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.output_elements; output_index += total_threads) {
    const int64_t input_base = ReduceInputBase(output_index, params);
    float acc = ReduceInitValue(op);
    for (int64_t r = 0; r < params.reduction_elements; ++r) {
      acc = ReduceUpdateValue(acc, input[input_base + ReduceInputOffset(r, params)], op);
    }
    if (op == MusaReduceOp::Mean) {
      acc /= static_cast<float>(params.reduction_elements);
    }
    output[output_index] = acc;
  }
}

__global__ void ReduceFloatBlockKernel(const float* input,
                                       float* output,
                                       MusaReduceParams params,
                                       MusaReduceOp op) {
  const int64_t output_index = static_cast<int64_t>(blockIdx.x);
  if (output_index >= params.output_elements) {
    return;
  }

  const int64_t input_base = ReduceInputBase(output_index, params);
  float acc = ReduceInitValue(op);
  for (int64_t r = threadIdx.x; r < params.reduction_elements; r += blockDim.x) {
    acc = ReduceUpdateValue(acc, input[input_base + ReduceInputOffset(r, params)], op);
  }

  __shared__ float shared[kThreadsPerBlock];
  shared[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = ReduceCombineValue(shared[threadIdx.x], shared[threadIdx.x + stride], op);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float value = shared[0];
    if (op == MusaReduceOp::Mean) {
      value /= static_cast<float>(params.reduction_elements);
    }
    output[output_index] = value;
  }
}

template <typename T>
__device__ __forceinline__ T ReduceIntInitValue(MusaReduceOp op) {
  return op == MusaReduceOp::Prod ? static_cast<T>(1) : static_cast<T>(0);
}

template <typename T>
__device__ __forceinline__ T ReduceIntUpdateValue(T acc, T value, MusaReduceOp op) {
  if (op == MusaReduceOp::Prod) {
    return acc * value;
  }
  if (op == MusaReduceOp::SumSquare) {
    return acc + value * value;
  }
  return acc + value;
}

template <typename T>
__device__ __forceinline__ T ReduceIntCombineValue(T lhs, T rhs, MusaReduceOp op) {
  if (op == MusaReduceOp::Prod) {
    return lhs * rhs;
  }
  return lhs + rhs;
}

template <typename T>
__global__ void ReduceIntKernel(const T* input,
                                T* output,
                                MusaReduceParams params,
                                MusaReduceOp op) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.output_elements; output_index += total_threads) {
    const int64_t input_base = ReduceInputBase(output_index, params);
    T acc = ReduceIntInitValue<T>(op);
    for (int64_t r = 0; r < params.reduction_elements; ++r) {
      acc = ReduceIntUpdateValue(acc, input[input_base + ReduceInputOffset(r, params)], op);
    }
    output[output_index] = acc;
  }
}

template <typename T>
__global__ void ReduceIntBlockKernel(const T* input,
                                     T* output,
                                     MusaReduceParams params,
                                     MusaReduceOp op) {
  const int64_t output_index = static_cast<int64_t>(blockIdx.x);
  if (output_index >= params.output_elements) return;
  const int64_t input_base = ReduceInputBase(output_index, params);
  T acc = ReduceIntInitValue<T>(op);
  for (int64_t r = threadIdx.x; r < params.reduction_elements; r += blockDim.x) {
    acc = ReduceIntUpdateValue(acc, input[input_base + ReduceInputOffset(r, params)], op);
  }
  __shared__ T shared[kThreadsPerBlock];
  shared[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = ReduceIntCombineValue(shared[threadIdx.x], shared[threadIdx.x + stride], op);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[output_index] = shared[0];
}

}  // namespace

musaError_t LaunchMusaReduceFloatKernel(const float* input,
                                        float* output,
                                        MusaReduceParams params,
                                        MusaReduceOp op,
                                        musaStream_t stream) {
  if (params.output_elements == 0) {
    return musaSuccess;
  }
  if (params.reduction_elements >= 1024 ||
      (params.reduction_elements >= 32 && params.output_elements <= 4096)) {
    ReduceFloatBlockKernel<<<static_cast<int>(params.output_elements), kThreadsPerBlock, 0, stream>>>(
        input, output, params, op);
    return musaGetLastError();
  }
  ReduceFloatKernel<<<BlocksForCount(params.output_elements), kThreadsPerBlock, 0, stream>>>(input, output, params, op);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchMusaReduceIntKernel(const T* input,
                                      T* output,
                                      MusaReduceParams params,
                                      MusaReduceOp op,
                                      musaStream_t stream) {
  if (params.output_elements == 0) return musaSuccess;
  if (op == MusaReduceOp::Mean) return musaErrorNotSupported;
  if (params.reduction_elements >= 1024 ||
      (params.reduction_elements >= 32 && params.output_elements <= 4096)) {
    ReduceIntBlockKernel<T><<<static_cast<int>(params.output_elements), kThreadsPerBlock, 0, stream>>>(
        input, output, params, op);
    return musaGetLastError();
  }
  ReduceIntKernel<T><<<BlocksForCount(params.output_elements), kThreadsPerBlock, 0, stream>>>(input, output, params, op);
  return musaGetLastError();
}

musaError_t LaunchMusaReduceInt32Kernel(const int32_t* input,
                                        int32_t* output,
                                        MusaReduceParams params,
                                        MusaReduceOp op,
                                        musaStream_t stream) {
  return LaunchMusaReduceIntKernel<int32_t>(input, output, params, op, stream);
}

musaError_t LaunchMusaReduceInt64Kernel(const int64_t* input,
                                        int64_t* output,
                                        MusaReduceParams params,
                                        MusaReduceOp op,
                                        musaStream_t stream) {
  return LaunchMusaReduceIntKernel<int64_t>(input, output, params, op, stream);
}
