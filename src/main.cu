#include <iostream>
#include <cuda_runtime.h>
#include <mlp_kernel.cuh>

int main() {
    const int input_size = 4;
    const int output_size = 3;

    float h_input[input_size] = {1.0f, -2.0f, 3.0f, 4.0f};

    // Host weights matrix (output_size x input_size)
    // Row-major: W[0] = weights for output neuron 0, etc.
    float h_weights[output_size * input_size] = {
        0.5f, -0.3f, 0.8f, 0.0f,   // Neuron 0
        -0.2f, 0.4f, 0.1f, 0.7f,   // Neuron 1
        0.3f, -0.5f, 0.2f, 0.9f    // Neuron 2
    };


    float h_bias[output_size] = {0.1f, -0.2f, 0.3f};

    float h_output[output_size] = {};

    // Device pointers
    float *d_input, *d_weights, *d_bias, *d_output;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_weights, output_size * input_size * sizeof(float));
    cudaMalloc(&d_bias, output_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, output_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel: one thread per output neuron
    int threads = 3;
    int blocks = (output_size + threads - 1) / threads;

    mlp_kernel<<<blocks, threads>>>(d_input, d_weights, d_bias, d_output, input_size, output_size);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output
    std::cout << "Linear + ReLU output:\n";
    for (int i = 0; i < output_size; ++i) {
        std::cout << "Neuron " << i << ": " << h_output[i] << "\n";
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);

    return 0;
}



