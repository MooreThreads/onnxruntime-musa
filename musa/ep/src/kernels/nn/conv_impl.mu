#include "nn/conv_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void Conv2DFloatKernel(const float* input,
                                  const float* weight,
                                  const float* bias,
                                  float* output,
                                  MusaConv2DParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < params.total_elements;
       index += total_threads) {
    int64_t remaining = index;
    const int64_t ow = remaining % params.out_w;
    remaining /= params.out_w;
    const int64_t oh = remaining % params.out_h;
    remaining /= params.out_h;
    const int64_t oc = remaining % params.m;
    remaining /= params.m;
    const int64_t n = remaining;

    float acc = bias == nullptr ? 0.0f : bias[oc];
    for (int64_t ic = 0; ic < params.c; ++ic) {
      for (int64_t kh = 0; kh < params.kernel_h; ++kh) {
        const int64_t ih =
            oh * params.stride_h + kh * params.dilation_h - params.pad_h;
        if (ih < 0 || ih >= params.h) {
          continue;
        }
        for (int64_t kw = 0; kw < params.kernel_w; ++kw) {
          const int64_t iw =
              ow * params.stride_w + kw * params.dilation_w - params.pad_w;
          if (iw < 0 || iw >= params.w) {
            continue;
          }
          const int64_t input_index =
              ((n * params.c + ic) * params.h + ih) * params.w + iw;
          const int64_t weight_index =
              ((oc * params.c + ic) * params.kernel_h + kh) *
                  params.kernel_w +
              kw;
          acc += input[input_index] * weight[weight_index];
        }
      }
    }
    output[index] = acc;
  }
}

}  // namespace

musaError_t LaunchMusaConv2DFloatKernel(const float* input,
                                        const float* weight,
                                        const float* bias,
                                        float* output,
                                        MusaConv2DParams params,
                                        musaStream_t stream) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }
  Conv2DFloatKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock,
                      0, stream>>>(input, weight, bias, output, params);
  return musaGetLastError();
}
