#include "reduction/reduction_functions.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

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
    int64_t remaining = output_index;
    int64_t input_base = 0;
    for (int32_t dim = 0; dim < params.rank; ++dim) {
      if (dim == params.reduce_axis) {
        continue;
      }
      const int64_t stride = params.output_strides[dim];
      const int64_t coord = stride == 0 ? 0 : remaining / stride;
      if (stride != 0) {
        remaining -= coord * stride;
      }
      input_base += coord * params.input_strides[dim];
    }

    float acc = ReduceInitValue(op);
    for (int64_t r = 0; r < params.reduce_dim; ++r) {
      acc = ReduceUpdateValue(acc, input[input_base + r * params.input_strides[params.reduce_axis]], op);
    }
    if (op == MusaReduceOp::Mean) {
      acc /= static_cast<float>(params.reduce_dim);
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

  int64_t remaining = output_index;
  int64_t input_base = 0;
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    if (dim == params.reduce_axis) {
      continue;
    }
    const int64_t stride = params.output_strides[dim];
    const int64_t coord = stride == 0 ? 0 : remaining / stride;
    if (stride != 0) {
      remaining -= coord * stride;
    }
    input_base += coord * params.input_strides[dim];
  }

  float acc = ReduceInitValue(op);
  for (int64_t r = threadIdx.x; r < params.reduce_dim; r += blockDim.x) {
    acc = ReduceUpdateValue(acc, input[input_base + r * params.input_strides[params.reduce_axis]], op);
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
      value /= static_cast<float>(params.reduce_dim);
    }
    output[output_index] = value;
  }
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
  if (params.reduce_dim >= 1024 ||
      (params.reduce_dim >= 32 && params.output_elements <= 4096)) {
    ReduceFloatBlockKernel<<<static_cast<int>(params.output_elements), kThreadsPerBlock, 0, stream>>>(
        input, output, params, op);
    musaError_t status = musaGetLastError();
    if (status != musaSuccess) {
      return status;
    }
    return musaSuccess;
  }
  ReduceFloatKernel<<<BlocksForCount(params.output_elements), kThreadsPerBlock, 0, stream>>>(input, output, params, op);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}
