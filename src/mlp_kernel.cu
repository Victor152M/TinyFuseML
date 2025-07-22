#include <mlp_kernel.cuh>
#include <relu.h>

__global__ void mlp_kernel(
    const float* __restrict__ input,      // [input_size]
    const float* __restrict__ weights,    // [output_size * input_size]
    const float* __restrict__ bias,       // [output_size]
    float* output,                        // [output_size]
    int input_size,
    int output_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += weights[idx * input_size + i] * input[i];
        }
        sum += bias[idx];
        output[idx] = relu(sum);
    }
}