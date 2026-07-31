// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include <musa_runtime.h>

#include <cstdint>

musaError_t LaunchMoEBiasReluFloatKernel(float* values, const float* biases,
                                         int64_t rows, int64_t expert_count,
                                         int64_t width, musaStream_t stream);

musaError_t LaunchMoERouterReduceMultiFloatKernel(
    const float* expert_values, const float* biases,
    const float* const* routers, float* const* outputs, int64_t rows,
    int64_t expert_count, int64_t output_width, int64_t router_count,
    musaStream_t stream);
