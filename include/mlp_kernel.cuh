#pragma once

__global__ void mlp_kernel(
    const float* __restrict__ input,      // [input_size]
    const float* __restrict__ weights,    // [output_size * input_size]
    const float* __restrict__ bias,       // [output_size]
    float* output,                        // [output_size]
    int input_size,
    int output_size);