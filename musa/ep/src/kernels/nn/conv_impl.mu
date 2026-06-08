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

__global__ void ConvHeightwise1C1MFloatKernel(const float* input,
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
    const int64_t n = remaining;

    float acc = bias == nullptr ? 0.0f : bias[0];
    for (int64_t kh = 0; kh < params.kernel_h; ++kh) {
      const int64_t ih = oh + kh - params.pad_h;
      if (ih >= 0 && ih < params.h) {
        acc += input[(n * params.h + ih) * params.w + ow] * weight[kh];
      }
    }
    output[index] = acc;
  }
}

__global__ void ConvHeightwise1C1MKernelH5Pad2SlidingFloatKernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    MusaConv2DParams params) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t column_count = params.n * params.w;
  const int64_t h = params.h;
  const int64_t w = params.w;
  const float k0 = weight[0];
  const float k1 = weight[1];
  const float k2 = weight[2];
  const float k3 = weight[3];
  const float k4 = weight[4];
  const float bias_value = bias == nullptr ? 0.0f : bias[0];

  for (int64_t column = thread_id; column < column_count;
       column += total_threads) {
    const int64_t n = column / w;
    const int64_t ow = column - n * w;
    const int64_t base = n * h * w + ow;

    float x0 = 0.0f;
    float x1 = 0.0f;
    float x2 = h > 0 ? input[base] : 0.0f;
    float x3 = h > 1 ? input[base + w] : 0.0f;
    float x4 = h > 2 ? input[base + 2 * w] : 0.0f;

    for (int64_t oh = 0; oh < h; ++oh) {
      output[base + oh * w] =
          bias_value + x0 * k0 + x1 * k1 + x2 * k2 + x3 * k3 + x4 * k4;
      x0 = x1;
      x1 = x2;
      x2 = x3;
      x3 = x4;
      const int64_t next_h = oh + 3;
      x4 = next_h < h ? input[base + next_h * w] : 0.0f;
    }
  }
}

bool CanUseHeightwise1C1MConv(MusaConv2DParams params) {
  return params.c == 1 && params.m == 1 && params.kernel_w == 1 &&
         params.out_w == params.w && params.pad_w == 0 &&
         params.stride_h == 1 && params.stride_w == 1 &&
         params.dilation_h == 1 && params.dilation_w == 1;
}

bool CanUseHeightwise1C1MKernelH5Pad2Conv(MusaConv2DParams params) {
  return CanUseHeightwise1C1MConv(params) && params.kernel_h == 5 &&
         params.pad_h == 2 && params.out_h == params.h;
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
  if (CanUseHeightwise1C1MKernelH5Pad2Conv(params)) {
    ConvHeightwise1C1MKernelH5Pad2SlidingFloatKernel
        <<<BlocksForCount(params.n * params.w), kThreadsPerBlock, 0, stream>>>(
            input, weight, bias, output, params);
    return musaGetLastError();
  }
  if (CanUseHeightwise1C1MConv(params)) {
    ConvHeightwise1C1MFloatKernel<<<BlocksForCount(params.total_elements),
                                    kThreadsPerBlock, 0, stream>>>(
        input, weight, bias, output, params);
    return musaGetLastError();
  }
  Conv2DFloatKernel<<<BlocksForCount(params.total_elements), kThreadsPerBlock,
                      0, stream>>>(input, weight, bias, output, params);
  return musaGetLastError();
}
