#include "nn/parallel_einsum_activation_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ const float* SelectHeadWeight(int64_t head,
                                                         const float* w0,
                                                         const float* w1,
                                                         const float* w2,
                                                         const float* w3) {
  switch (head) {
    case 0:
      return w0;
    case 1:
      return w1;
    case 2:
      return w2;
    default:
      return w3;
  }
}

__global__ void ParallelEinsumActivationPackW1Kernel(
    const float* w0, const float* w1, const float* w2, const float* w3,
    float* packed, int64_t input_dim, int64_t hidden_dim,
    int64_t total_elements) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < total_elements;
       index += total_threads) {
    int64_t remaining = index;
    const int64_t hidden = remaining % hidden_dim;
    remaining /= hidden_dim;
    const int64_t head = remaining % 4;
    remaining /= 4;
    const int64_t d = remaining;

    const float* weight = SelectHeadWeight(head, w0, w1, w2, w3);
    packed[d * (4 * hidden_dim) + head * hidden_dim + hidden] =
        weight[hidden * input_dim + d];
  }
}

__global__ void ParallelEinsumActivationStage23Kernel(
    const float* stage1, const float* gate_input, const float* w2_0,
    const float* w2_1, const float* w2_2, const float* w2_3, const float* w3_0,
    const float* w3_1, const float* w3_2, const float* w3_3, const float* bias,
    float* output, int64_t input_dim, int64_t hidden_dim) {
  extern __shared__ float h2[];

  const int64_t batch_index = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t tid = threadIdx.x;
  const float* h1 = stage1 + (batch_index * 4 + head) * hidden_dim;
  const float* gate = gate_input + batch_index * input_dim;
  const float* w2 = SelectHeadWeight(head, w2_0, w2_1, w2_2, w2_3);
  const float* w3 = SelectHeadWeight(head, w3_0, w3_1, w3_2, w3_3);

  for (int64_t hidden = tid; hidden < hidden_dim; hidden += blockDim.x) {
    float acc = 0.0f;
    const float* w = w2 + hidden * hidden_dim;
    for (int64_t k = 0; k < hidden_dim; ++k) {
      acc += w[k] * h1[k];
    }
    h2[hidden] = tanhf(acc);
  }
  __syncthreads();

  for (int64_t d = tid; d < input_dim; d += blockDim.x) {
    float acc = bias[d];
    const float* w = w3 + d * hidden_dim;
    for (int64_t k = 0; k < hidden_dim; ++k) {
      acc += w[k] * h2[k];
    }
    output[(batch_index * input_dim + d) * 4 + head] = gate[d] * acc;
  }
}

}  // namespace

musaError_t LaunchMusaParallelEinsumActivationPackW1Kernel(
    const float* w0, const float* w1, const float* w2, const float* w3,
    float* packed, int64_t input_dim, int64_t hidden_dim, musaStream_t stream) {
  const int64_t total_elements = input_dim * 4 * hidden_dim;
  if (total_elements == 0) {
    return musaSuccess;
  }
  ParallelEinsumActivationPackW1Kernel<<<BlocksForCount(total_elements),
                                         kThreadsPerBlock, 0, stream>>>(
      w0, w1, w2, w3, packed, input_dim, hidden_dim, total_elements);
  return musaGetLastError();
}

musaError_t LaunchMusaParallelEinsumActivationStage23Kernel(
    const float* stage1, const float* gate_input, const float* w2_0,
    const float* w2_1, const float* w2_2, const float* w2_3, const float* w3_0,
    const float* w3_1, const float* w3_2, const float* w3_3, const float* bias,
    float* output, int64_t batch, int64_t input_dim, int64_t hidden_dim,
    musaStream_t stream) {
  if (batch == 0 || input_dim == 0 || hidden_dim == 0) {
    return musaSuccess;
  }
  dim3 grid(batch, 4, 1);
  const size_t shared_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
  ParallelEinsumActivationStage23Kernel<<<grid, kThreadsPerBlock, shared_bytes,
                                          stream>>>(
      stage1, gate_input, w2_0, w2_1, w2_2, w2_3, w3_0, w3_1, w3_2, w3_3, bias,
      output, input_dim, hidden_dim);
  return musaGetLastError();
}
