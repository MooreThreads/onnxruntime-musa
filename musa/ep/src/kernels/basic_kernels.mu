#include "basic_kernels.h"

#include <math.h>
#include <stdint.h>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 4096;

int BlocksForCount(int64_t count) {
  int64_t blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (blocks > kMaxBlocks) {
    blocks = kMaxBlocks;
  }
  return static_cast<int>(blocks);
}

__device__ __forceinline__ void ResolveBroadcastIndices(int64_t index,
                                                        const MusaBroadcastParams& params,
                                                        int64_t& lhs_index,
                                                        int64_t& rhs_index) {
  lhs_index = 0;
  rhs_index = 0;
  int64_t remaining = index;
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    const int64_t coord = remaining / params.output_strides[dim];
    remaining -= coord * params.output_strides[dim];
    lhs_index += coord * params.lhs_strides[dim];
    rhs_index += coord * params.rhs_strides[dim];
  }
}

__device__ __forceinline__ float BinaryValue(float lhs, float rhs, MusaBinaryOp op) {
  switch (op) {
    case MusaBinaryOp::Add:
      return lhs + rhs;
    case MusaBinaryOp::Sub:
      return lhs - rhs;
    case MusaBinaryOp::Mul:
      return lhs * rhs;
    case MusaBinaryOp::Div:
      return lhs / rhs;
    case MusaBinaryOp::Pow:
      return powf(lhs, rhs);
    case MusaBinaryOp::Max:
      return lhs > rhs ? lhs : rhs;
    case MusaBinaryOp::Min:
      return lhs < rhs ? lhs : rhs;
  }
  return lhs;
}

__device__ __forceinline__ float UnaryValue(float x, MusaUnaryOp op, float alpha) {
  switch (op) {
    case MusaUnaryOp::Relu:
      return x > 0.0f ? x : 0.0f;
    case MusaUnaryOp::LeakyRelu:
      return x >= 0.0f ? x : alpha * x;
    case MusaUnaryOp::Sqrt:
      return sqrtf(x);
    case MusaUnaryOp::Reciprocal:
      return 1.0f / x;
    case MusaUnaryOp::Neg:
      return -x;
    case MusaUnaryOp::Log:
      return logf(x);
    case MusaUnaryOp::Tanh:
      return tanhf(x);
    case MusaUnaryOp::Sigmoid:
      return 1.0f / (1.0f + expf(-x));
    case MusaUnaryOp::Abs:
      return fabsf(x);
    case MusaUnaryOp::Erf:
      return erff(x);
  }
  return x;
}

template <typename T>
__device__ __forceinline__ uint8_t CompareValue(T lhs, T rhs, MusaCompareOp op) {
  switch (op) {
    case MusaCompareOp::Equal:
      return static_cast<uint8_t>(lhs == rhs);
    case MusaCompareOp::Greater:
      return static_cast<uint8_t>(lhs > rhs);
  }
  return 0;
}

__global__ void BinaryFloatKernel(const float* lhs,
                                  const float* rhs,
                                  float* output,
                                  MusaBroadcastParams params,
                                  MusaBinaryOp op) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    output[index] = BinaryValue(lhs[lhs_index], rhs[rhs_index], op);
  }
}

__global__ void UnaryFloatKernel(const float* input,
                                 float* output,
                                 int64_t count,
                                 MusaUnaryOp op,
                                 float alpha) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = UnaryValue(input[index], op, alpha);
  }
}

template <typename T>
__global__ void CompareKernel(const T* lhs,
                              const T* rhs,
                              uint8_t* output,
                              MusaBroadcastParams params,
                              MusaCompareOp op) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    output[index] = CompareValue(lhs[lhs_index], rhs[rhs_index], op);
  }
}

__global__ void NotBoolKernel(const uint8_t* input,
                              uint8_t* output,
                              int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = static_cast<uint8_t>(!input[index]);
  }
}

__global__ void OrBoolKernel(const uint8_t* lhs,
                             const uint8_t* rhs,
                             uint8_t* output,
                             MusaBroadcastParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, lhs_index, rhs_index);
    output[index] = static_cast<uint8_t>(lhs[lhs_index] || rhs[rhs_index]);
  }
}

__global__ void CastInt32ToFloatKernel(const int32_t* input,
                                       float* output,
                                       int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = static_cast<float>(input[index]);
  }
}

__global__ void CastInt64ToFloatKernel(const int64_t* input,
                                       float* output,
                                       int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = static_cast<float>(input[index]);
  }
}

__device__ __forceinline__ int64_t ReadGatherIndex(const void* indices,
                                                   int32_t index_element_size,
                                                   int64_t offset) {
  if (index_element_size == 4) {
    return static_cast<int64_t>(reinterpret_cast<const int32_t*>(indices)[offset]);
  }
  if (index_element_size == 8) {
    return reinterpret_cast<const int64_t*>(indices)[offset];
  }
  return 0;
}

__global__ void GatherKernel(const void* input,
                             const void* indices,
                             void* output,
                             int32_t element_size,
                             int32_t index_element_size,
                             int64_t input_block_size,
                             int64_t indices_max,
                             int64_t output_block_size,
                             int64_t block_size,
                             int64_t output_count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < output_count; output_index += total_threads) {
    const int64_t input_block_index = output_index / output_block_size;
    const int64_t block_offset = output_index % output_block_size;
    const int64_t indices_index = block_offset / block_size;
    const int64_t offset = block_offset % block_size;

    int64_t gather_index = ReadGatherIndex(indices, index_element_size, indices_index);
    if (gather_index < 0) {
      gather_index += indices_max;
    }

    if (gather_index < 0 || gather_index >= indices_max) {
      if (element_size == 4) {
        reinterpret_cast<uint32_t*>(output)[output_index] = 0;
      } else if (element_size == 8) {
        reinterpret_cast<uint64_t*>(output)[output_index] = 0;
      } else if (element_size == 1) {
        reinterpret_cast<uint8_t*>(output)[output_index] = 0;
      } else {
        uint8_t* dst = reinterpret_cast<uint8_t*>(output) + output_index * element_size;
        for (int32_t byte = 0; byte < element_size; ++byte) {
          dst[byte] = 0;
        }
      }
      continue;
    }

    const int64_t input_index = input_block_index * input_block_size + gather_index * block_size + offset;
    if (element_size == 4) {
      reinterpret_cast<uint32_t*>(output)[output_index] = reinterpret_cast<const uint32_t*>(input)[input_index];
    } else if (element_size == 8) {
      reinterpret_cast<uint64_t*>(output)[output_index] = reinterpret_cast<const uint64_t*>(input)[input_index];
    } else if (element_size == 1) {
      reinterpret_cast<uint8_t*>(output)[output_index] = reinterpret_cast<const uint8_t*>(input)[input_index];
    } else {
      const uint8_t* src = reinterpret_cast<const uint8_t*>(input) + input_index * element_size;
      uint8_t* dst = reinterpret_cast<uint8_t*>(output) + output_index * element_size;
      for (int32_t byte = 0; byte < element_size; ++byte) {
        dst[byte] = src[byte];
      }
    }
  }
}

__global__ void SliceKernel(const void* input,
                            void* output,
                            int32_t element_size,
                            MusaSliceParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.total_elements; output_index += total_threads) {
    int64_t remaining = output_index;
    int64_t input_index = 0;
    for (int32_t dim = params.rank - 1; dim >= 0; --dim) {
      const int64_t coord = remaining % params.output_dims[dim];
      remaining /= params.output_dims[dim];
      input_index += (params.starts[dim] + coord * params.steps[dim]) * params.input_strides[dim];
    }

    if (element_size == 4) {
      reinterpret_cast<uint32_t*>(output)[output_index] = reinterpret_cast<const uint32_t*>(input)[input_index];
    } else if (element_size == 8) {
      reinterpret_cast<uint64_t*>(output)[output_index] = reinterpret_cast<const uint64_t*>(input)[input_index];
    } else if (element_size == 1) {
      reinterpret_cast<uint8_t*>(output)[output_index] = reinterpret_cast<const uint8_t*>(input)[input_index];
    } else {
      const uint8_t* src = reinterpret_cast<const uint8_t*>(input) + input_index * element_size;
      uint8_t* dst = reinterpret_cast<uint8_t*>(output) + output_index * element_size;
      for (int32_t byte = 0; byte < element_size; ++byte) {
        dst[byte] = src[byte];
      }
    }
  }
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

__global__ void TransposeKernel(const void* input,
                                void* output,
                                int32_t element_size,
                                MusaTransposeParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.total_elements; output_index += total_threads) {
    int64_t remaining = output_index;
    int64_t input_index = 0;
    for (int32_t dim = params.rank - 1; dim >= 0; --dim) {
      const int64_t coord = remaining % params.output_dims[dim];
      remaining /= params.output_dims[dim];
      input_index += coord * params.input_strides[params.perm[dim]];
    }

    if (element_size == 4) {
      reinterpret_cast<uint32_t*>(output)[output_index] = reinterpret_cast<const uint32_t*>(input)[input_index];
    } else if (element_size == 8) {
      reinterpret_cast<uint64_t*>(output)[output_index] = reinterpret_cast<const uint64_t*>(input)[input_index];
    } else if (element_size == 1) {
      reinterpret_cast<uint8_t*>(output)[output_index] = reinterpret_cast<const uint8_t*>(input)[input_index];
    } else {
      const uint8_t* src = reinterpret_cast<const uint8_t*>(input) + input_index * element_size;
      uint8_t* dst = reinterpret_cast<uint8_t*>(output) + output_index * element_size;
      for (int32_t byte = 0; byte < element_size; ++byte) {
        dst[byte] = src[byte];
      }
    }
  }
}

__global__ void BatchNormalizationFloatKernel(const float* input,
                                              const float* scale,
                                              const float* bias,
                                              const float* mean,
                                              const float* variance,
                                              float* output,
                                              MusaBatchNormParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    const int64_t channel = params.channels == 1 ? 0 : (index / params.spatial_size) % params.channels;
    output[index] = (input[index] - mean[channel]) * rsqrtf(variance[channel] + params.epsilon) *
                    scale[channel] + bias[channel];
  }
}

__global__ void SoftmaxFloatKernel(const float* input,
                                   float* output,
                                   int64_t rows,
                                   int64_t dim,
                                   int64_t inner) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t row_index = thread_id; row_index < rows; row_index += total_threads) {
    const int64_t outer_index = row_index / inner;
    const int64_t inner_index = row_index - outer_index * inner;
    const int64_t base = outer_index * dim * inner + inner_index;

    float max_value = -INFINITY;
    for (int64_t d = 0; d < dim; ++d) {
      const float value = input[base + d * inner];
      max_value = value > max_value ? value : max_value;
    }

    float sum = 0.0f;
    for (int64_t d = 0; d < dim; ++d) {
      const float value = expf(input[base + d * inner] - max_value);
      output[base + d * inner] = value;
      sum += value;
    }

    const float inv_sum = 1.0f / sum;
    for (int64_t d = 0; d < dim; ++d) {
      output[base + d * inner] *= inv_sum;
    }
  }
}

}  // namespace

musaError_t LaunchMusaBinaryFloatKernel(const float* lhs,
                                        const float* rhs,
                                        float* output,
                                        MusaBroadcastParams params,
                                        MusaBinaryOp op,
                                        musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  BinaryFloatKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(lhs, rhs, output, params, op);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}

musaError_t LaunchMusaUnaryFloatKernel(const float* input,
                                       float* output,
                                       int64_t count,
                                       MusaUnaryOp op,
                                       float alpha,
                                       musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  UnaryFloatKernel<<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(input, output, count, op, alpha);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}

musaError_t LaunchMusaCompareFloatKernel(const float* lhs,
                                         const float* rhs,
                                         uint8_t* output,
                                         MusaBroadcastParams params,
                                         MusaCompareOp op,
                                         musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  CompareKernel<float><<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(lhs, rhs, output, params, op);
  return musaGetLastError();
}

musaError_t LaunchMusaCompareInt32Kernel(const int32_t* lhs,
                                         const int32_t* rhs,
                                         uint8_t* output,
                                         MusaBroadcastParams params,
                                         MusaCompareOp op,
                                         musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  CompareKernel<int32_t><<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(lhs, rhs, output, params, op);
  return musaGetLastError();
}

musaError_t LaunchMusaCompareInt64Kernel(const int64_t* lhs,
                                         const int64_t* rhs,
                                         uint8_t* output,
                                         MusaBroadcastParams params,
                                         MusaCompareOp op,
                                         musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  CompareKernel<int64_t><<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(lhs, rhs, output, params, op);
  return musaGetLastError();
}

musaError_t LaunchMusaNotBoolKernel(const uint8_t* input,
                                    uint8_t* output,
                                    int64_t count,
                                    musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  NotBoolKernel<<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(input, output, count);
  return musaGetLastError();
}

musaError_t LaunchMusaOrBoolKernel(const uint8_t* lhs,
                                   const uint8_t* rhs,
                                   uint8_t* output,
                                   MusaBroadcastParams params,
                                   musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  OrBoolKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(lhs, rhs, output, params);
  return musaGetLastError();
}

musaError_t LaunchMusaCastInt32ToFloatKernel(const int32_t* input,
                                             float* output,
                                             int64_t count,
                                             musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  CastInt32ToFloatKernel<<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(input, output, count);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}

musaError_t LaunchMusaCastInt64ToFloatKernel(const int64_t* input,
                                             float* output,
                                             int64_t count,
                                             musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  CastInt64ToFloatKernel<<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(input, output, count);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}

musaError_t LaunchMusaGatherKernel(const void* input,
                                   const void* indices,
                                   void* output,
                                   int32_t element_size,
                                   int32_t index_element_size,
                                   int64_t input_block_size,
                                   int64_t indices_max,
                                   int64_t output_block_size,
                                   int64_t block_size,
                                   int64_t output_count,
                                   musaStream_t stream) {
  if (output_count == 0) {
    return musaSuccess;
  }
  GatherKernel<<<BlocksForCount(output_count), kThreadsPerBlock, 0, stream>>>(
      input, indices, output, element_size, index_element_size, input_block_size,
      indices_max, output_block_size, block_size, output_count);
  return musaGetLastError();
}

musaError_t LaunchMusaSliceKernel(const void* input,
                                  void* output,
                                  int32_t element_size,
                                  MusaSliceParams params,
                                  musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  SliceKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(input, output, element_size, params);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}

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

musaError_t LaunchMusaBatchNormalizationFloatKernel(const float* input,
                                                    const float* scale,
                                                    const float* bias,
                                                    const float* mean,
                                                    const float* variance,
                                                    float* output,
                                                    MusaBatchNormParams params,
                                                    musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  BatchNormalizationFloatKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
      input, scale, bias, mean, variance, output, params);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}

musaError_t LaunchMusaTransposeKernel(const void* input,
                                      void* output,
                                      int32_t element_size,
                                      MusaTransposeParams params,
                                      musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  TransposeKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0, stream>>>(
      input, output, element_size, params);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}

musaError_t LaunchMusaSoftmaxFloatKernel(const float* input,
                                         float* output,
                                         int64_t outer,
                                         int64_t dim,
                                         int64_t inner,
                                         musaStream_t stream) {
  const int64_t rows = outer * inner;
  if (rows == 0 || dim == 0) {
    return musaSuccess;
  }
  SoftmaxFloatKernel<<<BlocksForCount(rows), kThreadsPerBlock, 0, stream>>>(input, output, rows, dim, inner);
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}
