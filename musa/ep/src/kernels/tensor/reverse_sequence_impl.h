#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaReverseSequenceKernel(const void* input,
                                            const int64_t* sequence_lens,
                                            void* output, int32_t element_size,
                                            MusaReverseSequenceParams params,
                                            musaStream_t stream);
