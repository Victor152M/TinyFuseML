#include <mlp_kernel.cuh>
#include <relu.h>

__global__ void mlp_kernel(
    const float* __restrict__ input,
    float* __restrict__ weightsLayer1,
    float* __restrict__ biasLayer1,
    float* __restrict__ weightsLayer2,
    float* __restrict__ biasLayer2,
    float* output,
    int input_size,
    int hidden_size,
    int output_size,
    float learning_rate,
    const float* __restrict__ target,
    bool training)
{
    //Forward pass

    //Hopefully stored in GPU registers
    float intermediateActivation[3];

    #pragma unroll
    for (int i = 0; i < 3; i++){
        float sum = biasLayer1[i];
        sum += weightsLayer1[i * 3 + 0] * input[0];
        sum += weightsLayer1[i * 3 + 1] * input[1];
        sum += weightsLayer1[i * 3 + 2] * input[2];
        intermediateActivation[i] = relu(sum);
    }

    #pragma unroll
    for (int i = 0; i < 3; i++){
        float sum = biasLayer2[i];
        sum += weightsLayer2[i * 3 + 0] * intermediateActivation[0];
        sum += weightsLayer2[i * 3 + 1] * intermediateActivation[1];
        sum += weightsLayer2[i * 3 + 2] * intermediateActivation[2];
        output[i] = sum; // Normally relu(sum) but just sum for regression
    }

    if (!training) return;

    // Backward pass

    // Graadient of the loss
    float outputGradient[3];

    #pragma unroll
    for (int i = 0; i < 3; i++){
        // Derivative of the Mean Squared Error
        outputGradient[i] = output[i] - target[i];
    }

    // Gradients for weightsLayer2 (dL/dW2)
    float preActivationLayer2_gradient[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        //preActivationLayer2_gradient[i] = outputGradient[i] * reluDerivative(output[i]); before removing relu from the output
        preActivationLayer2_gradient[i] = outputGradient[i];
    }

    // Updating Layer 2
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        int row = i / 3;
        int col = i % 3;
        float grad = preActivationLayer2_gradient[row] * intermediateActivation[col];
        weightsLayer2[i] -= learning_rate * grad;
    }

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        biasLayer2[i] -= learning_rate * preActivationLayer2_gradient[i];
    }

    //Layer 1
    float layer1_output_gradient[3];

    layer1_output_gradient[0] =
        (preActivationLayer2_gradient[0] * weightsLayer2[0 * 3 + 0]) +
        (preActivationLayer2_gradient[1] * weightsLayer2[1 * 3 + 0]) +
        (preActivationLayer2_gradient[2] * weightsLayer2[2 * 3 + 0]);

    layer1_output_gradient[1] =
        (preActivationLayer2_gradient[0] * weightsLayer2[0 * 3 + 1]) +
        (preActivationLayer2_gradient[1] * weightsLayer2[1 * 3 + 1]) +
        (preActivationLayer2_gradient[2] * weightsLayer2[2 * 3 + 1]);

    layer1_output_gradient[2] =
        (preActivationLayer2_gradient[0] * weightsLayer2[0 * 3 + 2]) +
        (preActivationLayer2_gradient[1] * weightsLayer2[1 * 3 + 2]) +
        (preActivationLayer2_gradient[2] * weightsLayer2[2 * 3 + 2]);

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        layer1_output_gradient[i] *= reluDerivative(intermediateActivation[i]);
    }

    // Updating Layer 1
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        int row = i / 3;
        int col = i % 3;
        float grad = layer1_output_gradient[row] * input[col];
        weightsLayer1[i] -= learning_rate * grad;
    }

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        biasLayer1[i] -= learning_rate * layer1_output_gradient[i];
    }

}

