#include "matmul_batched_kernels.h"
#include "shared_inc/musa_kernel_common.mu.h"

#include <stdint.h>

namespace {

template <typename T, typename AccT>
__global__ void BatchedMatMulKernel(const T* a,
                                    const T* b,
                                    T* output,
                                    MusaBatchedMatMulParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t output_index = thread_id; output_index < params.total_elements;
       output_index += total_threads) {
    int64_t remaining = output_index;
    int64_t a_base = 0;
    int64_t b_base = 0;
    int64_t row = 0;
    int64_t col = 0;
    for (int32_t dim = 0; dim < params.output_rank; ++dim) {
      const int64_t coord = remaining / params.output_strides[dim];
      remaining -= coord * params.output_strides[dim];
      if (dim < params.batch_rank) {
        a_base += coord * params.a_batch_strides[dim];
        b_base += coord * params.b_batch_strides[dim];
      } else if (dim == params.batch_rank) {
        row = coord;
      } else {
        col = coord;
      }
    }

    AccT sum = static_cast<AccT>(0);
    for (int64_t kk = 0; kk < params.k; ++kk) {
      sum += static_cast<AccT>(MusaScalarToDouble(
                 a[a_base + row * params.a_row_stride +
                   kk * params.a_col_stride])) *
             static_cast<AccT>(MusaScalarToDouble(
                 b[b_base + kk * params.b_row_stride +
                   col * params.b_col_stride]));
    }
    output[output_index] =
        MusaScalarFromDouble<T>(static_cast<double>(params.alpha) *
                                static_cast<double>(sum));
  }
}

__global__ void BatchedDotFloatKernel(const float* a,
                                      const float* b,
                                      float* output,
                                      MusaBatchedMatMulParams params) {
  const int64_t output_index = static_cast<int64_t>(blockIdx.x);
  if (output_index >= params.total_elements) {
    return;
  }

  int64_t remaining = output_index;
  int64_t a_base = 0;
  int64_t b_base = 0;
  for (int32_t dim = 0; dim < params.batch_rank; ++dim) {
    const int64_t coord = remaining / params.output_strides[dim];
    remaining -= coord * params.output_strides[dim];
    a_base += coord * params.a_batch_strides[dim];
    b_base += coord * params.b_batch_strides[dim];
  }

  float sum = 0.0f;
  for (int64_t kk = threadIdx.x; kk < params.k; kk += blockDim.x) {
    sum += a[a_base + kk * params.a_col_stride] *
           b[b_base + kk * params.b_row_stride];
  }

  __shared__ float shared[kThreadsPerBlock];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    output[output_index] = params.alpha * shared[0];
  }
}

}  // namespace

musaError_t LaunchMusaBatchedMatMulKernel(const void* a, const void* b,
                                          void* output,
                                          MusaBatchedMatMulParams params,
                                          MusaElementType elem_type,
                                          musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  switch (elem_type) {
    case MusaElementType::Float:
      BatchedMatMulKernel<float, float>
          <<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0,
             stream>>>(reinterpret_cast<const float*>(a),
                       reinterpret_cast<const float*>(b),
                       reinterpret_cast<float*>(output), params);
      break;
    case MusaElementType::Double:
      BatchedMatMulKernel<double, double>
          <<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0,
             stream>>>(reinterpret_cast<const double*>(a),
                       reinterpret_cast<const double*>(b),
                       reinterpret_cast<double*>(output), params);
      break;
    case MusaElementType::Float16:
      BatchedMatMulKernel<__half, float>
          <<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0,
             stream>>>(reinterpret_cast<const __half*>(a),
                       reinterpret_cast<const __half*>(b),
                       reinterpret_cast<__half*>(output), params);
      break;
    case MusaElementType::BFloat16:
      BatchedMatMulKernel<__mt_bfloat16, float>
          <<<BlocksForCount(params.total_elements), kThreadsPerBlock, 0,
             stream>>>(reinterpret_cast<const __mt_bfloat16*>(a),
                       reinterpret_cast<const __mt_bfloat16*>(b),
                       reinterpret_cast<__mt_bfloat16*>(output), params);
      break;
    default:
      return musaErrorNotSupported;
  }
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    return status;
  }
  return musaSuccess;
}

musaError_t LaunchMusaBatchedDotFloatKernel(const void* a, const void* b,
                                            void* output,
                                            MusaBatchedMatMulParams params,
                                            musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  if (params.m != 1 || params.n != 1 || params.k <= 0) {
    return musaErrorNotSupported;
  }
  BatchedDotFloatKernel<<<static_cast<int>(params.total_elements),
                          kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const float*>(a), reinterpret_cast<const float*>(b),
      reinterpret_cast<float*>(output), params);
  return musaGetLastError();
}
