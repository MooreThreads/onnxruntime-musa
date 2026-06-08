#pragma once

#include <musa_runtime.h>

#include <cstdint>

struct MusaShapeGatherParams {
  int64_t dims[8];
  int64_t indices[8];
  int32_t output_count;
  int32_t rank;
  int32_t output_type;
};

musaError_t LaunchMusaShapeGatherKernel(void* output,
                                        MusaShapeGatherParams params,
                                        musaStream_t stream);
