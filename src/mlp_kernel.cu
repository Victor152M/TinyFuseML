#include <relu.h>
#include <hash_encoding.cuh>
#include <cstdio>


__global__ void MLPKernel(
    const float* __restrict__ positions,  // raw x,y pairs
    float* __restrict__ weightsLayer1,
    float* __restrict__ biasLayer1,
    float* __restrict__ weightsLayer2,
    float* __restrict__ biasLayer2,
    float* __restrict__ weightsLayer3, 
    float* __restrict__ biasLayer3,
    float* __restrict__ outputs,
    int inputSize,
    int hiddenSize1,
    int hiddenSize2,
    int outputSize,
    int batchSize,
    float learningRate,
    float hashLearningRate,
    const float* __restrict__ targets,
    bool training,
    float* __restrict__ hashTable,
    int baseHashResolution
)
{
    // Identify the sample each thread works on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    float encodedInput[HASH_ENCODED_SIZE];
    int hashIndices[HASH_ENCODED_SIZE * 4];

    float x = positions[idx * 2 + 0];
    float y = positions[idx * 2 + 1];

    HashEncode(x, y, encodedInput, hashTable, hashIndices, baseHashResolution);

    const float* input = encodedInput;
    const float* target = targets + idx * outputSize;
    float* output = outputs + idx * outputSize;

    extern __shared__ float shared[];

    int tid = threadIdx.x;

    int local_size = hiddenSize1 + hiddenSize2 + outputSize + outputSize + hiddenSize2 + hiddenSize1;
    float* activation1                 = shared + tid * local_size;
    float* activation2                 = activation1 + hiddenSize1;
    float* outputGradient              = activation2 + hiddenSize2;
    float* preActivationGrad3          = outputGradient + outputSize;
    float* layer2_output_gradient      = preActivationGrad3 + outputSize;
    float* layer1_output_gradient      = layer2_output_gradient + hiddenSize2;

    //Forward pass
    #pragma unroll
    for (int i = 0; i < hiddenSize1; i++) 
    {
        float sum = biasLayer1[i];
        for (int j = 0; j < inputSize; j++) 
            sum += weightsLayer1[i * inputSize + j] * input[j];
        activation1[i] = Relu(sum);
    }

    #pragma unroll
    // Hidden1 to Hidden2
    for (int i = 0; i < hiddenSize2; i++) 
    {
        float sum = biasLayer2[i];
        for (int j = 0; j < hiddenSize1; j++)
            sum += weightsLayer2[i * hiddenSize1 + j] * activation1[j];
        activation2[i] = Relu(sum);
    }

    // Hidden2 to Output (Linear)
    for (int i = 0; i < outputSize; i++) 
    {
        float sum = biasLayer3[i];
        for (int j = 0; j < hiddenSize2; j++)
            sum += weightsLayer3[i * hiddenSize2 + j] * activation2[j];
        output[i] = sum;
    }

    if (!training) return;

    // Backward pass

    // Gradient of the loss
    for (int i = 0; i < outputSize; i++) {
        outputGradient[i]     = output[i] - target[i];
        preActivationGrad3[i] = outputGradient[i]; // Linear output
    }
    
    // Update weights for Layer3: Hidden2 to Output
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < hiddenSize2; j++) {
            float grad = preActivationGrad3[i] * activation2[j];
            grad = fminf(fmaxf(grad, -1.0f), 1.0f);
            atomicAdd(&weightsLayer3[i * hiddenSize2 + j], -learningRate * grad);
        }
        atomicAdd(&biasLayer3[i], -learningRate * preActivationGrad3[i]);
    }

    // Backprop: Layer3 to Layer2
    for (int j = 0; j < hiddenSize2; j++) {
        float grad = 0.0f;
        for (int i = 0; i < outputSize; i++) {
            grad += preActivationGrad3[i] * weightsLayer3[i * hiddenSize2 + j];
        }
        layer2_output_gradient[j] = grad * ReluDerivative(activation2[j]);
    }

    #pragma unroll
    // Update weights for Layer2: Hidden1 to Hidden2
    for (int i = 0; i < hiddenSize2; i++) {
        for (int j = 0; j < hiddenSize1; j++) {
            float grad = layer2_output_gradient[i] * activation1[j];
            grad = fminf(fmaxf(grad, -1.0f), 1.0f);
            atomicAdd(&weightsLayer2[i * hiddenSize1 + j], -learningRate * grad);
        }
        atomicAdd(&biasLayer2[i], -learningRate * layer2_output_gradient[i]);
    }

        // Backprop: Layer2 to Layer1
    for (int j = 0; j < hiddenSize1; j++) {
        float grad = 0.0f;
        for (int i = 0; i < hiddenSize2; i++) {
            grad += layer2_output_gradient[i] * weightsLayer2[i * hiddenSize1 + j];
        }
        layer1_output_gradient[j] = grad * ReluDerivative(activation1[j]);
    }


    // Backprop into hashTable using layer1_output_gradient
    int hashOffset = 0;
    float base_lr = hashLearningRate;
    float base = 1.4f;
    
    for (int level = 0; level < N_LEVELS; ++level) {
        //float hash_lr = 0.9f / powf(SCALE_FACTOR, level);
        //float hash_lr = hashLearningRate;
        float hash_lr = hashLearningRate;
        int resolution = static_cast<int>(baseHashResolution * powf(SCALE_FACTOR, level));

        float fx = x * resolution;
        float fy = y * resolution;
        int x0 = floorf(fx);
        int y0 = floorf(fy);
        float dx = fx - x0;
        float dy = fy - y0;

        dx = fminf(fmaxf(dx, 0.0f), 1.0f);
        dy = fminf(fmaxf(dy, 0.0f), 1.0f);

        for (int f = 0; f < FEATURES_PER_LEVEL; ++f) {
            int idx00 = hashIndices[hashOffset++];
            int idx10 = hashIndices[hashOffset++];
            int idx01 = hashIndices[hashOffset++];
            int idx11 = hashIndices[hashOffset++];

            float w00 = (1 - dx) * (1 - dy);
            float w10 = dx * (1 - dy);
            float w01 = (1 - dx) * dy;
            float w11 = dx * dy;

            float grad = 0.0f;
            for (int j = 0; j < hiddenSize1; ++j) {
                grad += weightsLayer1[j * inputSize + level * FEATURES_PER_LEVEL + f] * layer1_output_gradient[j];
            }
            //grad = fminf(fmaxf(grad, -1.0f), 1.0f);

            atomicAdd(&hashTable[idx00], -hash_lr * w00 * grad);
            atomicAdd(&hashTable[idx10], -hash_lr * w10 * grad);
            atomicAdd(&hashTable[idx01], -hash_lr * w01 * grad);
            atomicAdd(&hashTable[idx11], -hash_lr * w11 * grad);
        }
    }


    // Update weights for Layer1: Input to Hidden1
    for (int i = 0; i < hiddenSize1; i++) {
        for (int j = 0; j < inputSize; j++) {
            float grad = layer1_output_gradient[i] * input[j];
            grad = fminf(fmaxf(grad, -1.0f), 1.0f);
            atomicAdd(&weightsLayer1[i * inputSize + j], -learningRate * grad);
        }
        atomicAdd(&biasLayer1[i], -learningRate * layer1_output_gradient[i]);
    }
}

__global__ void MLPSDFKernel(
    const float* __restrict__ sdfPositions, // x, y, z
    float* __restrict__ weightsLayer1,
    float* __restrict__ biasLayer1,
    float* __restrict__ weightsLayer2,
    float* __restrict__ biasLayer2,
    float* __restrict__ weightsLayer3,
    float* __restrict__ biasLayer3,
    float* __restrict__ outputs,
    int inputSize,
    int hiddenSize1,
    int hiddenSize2,
    int outputSize,
    int batchSize,
    float learningRate,
    float hashLearningRate,
    const float* __restrict__ targets,
    bool training,
    float* __restrict__ hashTable,
    int baseHashResolution
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    float encodedInput[HASH_ENCODED_SIZE];
    int hashIndices[HASH_ENCODED_SIZE * 8]; // 3D

    float x = sdfPositions[idx * 3 + 0];
    float y = sdfPositions[idx * 3 + 1];
    float z = sdfPositions[idx * 3 + 2];


    HashEncode3D(x, y, z, encodedInput, hashTable, hashIndices, baseHashResolution);

    const float* input = encodedInput;
    const float* target = targets + idx * outputSize;
    float* output = outputs + idx * outputSize;

    extern __shared__ float shared[];

    int tid = threadIdx.x;

    int local_size = hiddenSize1 + hiddenSize2 + outputSize + outputSize + hiddenSize2 + hiddenSize1;
    float* activation1                 = shared + tid * local_size;
    float* activation2                 = activation1 + hiddenSize1;
    float* outputGradient              = activation2 + hiddenSize2;
    float* preActivationGrad3          = outputGradient + outputSize;
    float* layer2_output_gradient      = preActivationGrad3 + outputSize;
    float* layer1_output_gradient      = layer2_output_gradient + hiddenSize2;

    //Forward pass
    #pragma unroll
    for (int i = 0; i < hiddenSize1; i++) {
        float sum = biasLayer1[i];
        for (int j = 0; j < inputSize; j++) {
            sum += weightsLayer1[i * inputSize + j] * input[j];
        }
        activation1[i] = Relu(sum);
    }

    #pragma unroll
    // Hidden1 to Hidden2
    for (int i = 0; i < hiddenSize2; i++) {
        float sum = biasLayer2[i];
        for (int j = 0; j < hiddenSize1; j++) {
            sum += weightsLayer2[i * hiddenSize1 + j] * activation1[j];
        }
        activation2[i] = Relu(sum);
    }

    // Hidden2 to Output (Linear)
    for (int i = 0; i < outputSize; i++) {
        float sum = biasLayer3[i];
        for (int j = 0; j < hiddenSize2; j++) {
            sum += weightsLayer3[i * hiddenSize2 + j] * activation2[j];
        }
        output[i] = sum;
    }

    if (!training) return;

    // Backward pass

    // Gradient of the loss 
    for (int i = 0; i < outputSize; i++) {
        outputGradient[i]     = output[i] - target[i];
        preActivationGrad3[i] = outputGradient[i]; // Linear output
    }

    // Update weights for Layer3: Hidden2 to Output
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < hiddenSize2; j++) {
            float grad = preActivationGrad3[i] * activation2[j];
            grad = fminf(fmaxf(grad, -1.0f), 1.0f);
            atomicAdd(&weightsLayer3[i * hiddenSize2 + j], -learningRate * grad);
        }
        atomicAdd(&biasLayer3[i], -learningRate * preActivationGrad3[i]);
    }

   // Backprop: Layer3 to Layer2
    for (int j = 0; j < hiddenSize2; j++) {
        float grad = 0.0f;
        for (int i = 0; i < outputSize; i++) {
            grad += preActivationGrad3[i] * weightsLayer3[i * hiddenSize2 + j];
        }
        layer2_output_gradient[j] = grad * ReluDerivative(activation2[j]);
    }

    #pragma unroll
    // Update weights for Layer2: Hidden1 to Hidden2
    for (int i = 0; i < hiddenSize2; i++) {
        for (int j = 0; j < hiddenSize1; j++) {
            float grad = layer2_output_gradient[i] * activation1[j];
            grad = fminf(fmaxf(grad, -1.0f), 1.0f);
            atomicAdd(&weightsLayer2[i * hiddenSize1 + j], -learningRate * grad);
        }
        atomicAdd(&biasLayer2[i], -learningRate * layer2_output_gradient[i]);
    }

    // Backprop: Layer2 to Layer1
    for (int j = 0; j < hiddenSize1; j++) {
        float grad = 0.0f;
        for (int i = 0; i < hiddenSize2; i++) {
            grad += layer2_output_gradient[i] * weightsLayer2[i * hiddenSize1 + j];
        }
        layer1_output_gradient[j] = grad * ReluDerivative(activation1[j]);
    }

    // Backprop into hashTable using layer1_output_gradient 
    int hashOffset = 0;
    for (int level = 0; level < N_LEVELS; ++level) {
        float base_lr = hashLearningRate;
        float base = 1.2f;
        // float hash_lr = base_lr / powf(base, level);
        float hash_lr = hashLearningRate;
        int resolution = static_cast<int>(baseHashResolution * powf(SCALE_FACTOR, level));

        float fx = x * resolution;
        float fy = y * resolution;
        float fz = z * resolution;
        int x0 = floorf(fx);
        int y0 = floorf(fy);
        int z0 = floorf(fz);
        float dx = fx - x0;
        float dy = fy - y0;
        float dz = fz - z0;

        for (int f = 0; f < FEATURES_PER_LEVEL; ++f) {
            dx = fminf(fmaxf(dx, 0.0f), 1.0f);
            dy = fminf(fmaxf(dy, 0.0f), 1.0f);
            dz = fminf(fmaxf(dz, 0.0f), 1.0f);

            int idx000 = hashIndices[hashOffset++];
            int idx100 = hashIndices[hashOffset++];
            int idx010 = hashIndices[hashOffset++];
            int idx110 = hashIndices[hashOffset++];
            int idx001 = hashIndices[hashOffset++];
            int idx101 = hashIndices[hashOffset++];
            int idx011 = hashIndices[hashOffset++];
            int idx111 = hashIndices[hashOffset++];

            float w000 = (1-dx)*(1-dy)*(1-dz);
            float w100 = dx*(1-dy)*(1-dz);
            float w010 = (1-dx)*dy*(1-dz);
            float w110 = dx*dy*(1-dz);
            float w001 = (1-dx)*(1-dy)*dz;
            float w101 = dx*(1-dy)*dz;
            float w011 = (1-dx)*dy*dz;
            float w111 = dx*dy*dz;

            float grad = 0.0f;
            for (int j = 0; j < hiddenSize1; ++j) {
                grad += weightsLayer1[j * inputSize + level * FEATURES_PER_LEVEL + f] * layer1_output_gradient[j];
            }

            grad = fminf(fmaxf(grad, -1.0f), 1.0f);
            if (!isfinite(grad)) grad = 0.0f;

            // distribute to the 8 hashed storage entries using trilinear weights
            atomicAdd(&hashTable[idx000], -hash_lr * w000 * grad);
            atomicAdd(&hashTable[idx100], -hash_lr * w100 * grad);
            atomicAdd(&hashTable[idx010], -hash_lr * w010 * grad);
            atomicAdd(&hashTable[idx110], -hash_lr * w110 * grad);
            atomicAdd(&hashTable[idx001], -hash_lr * w001 * grad);
            atomicAdd(&hashTable[idx101], -hash_lr * w101 * grad);
            atomicAdd(&hashTable[idx011], -hash_lr * w011 * grad);
            atomicAdd(&hashTable[idx111], -hash_lr * w111 * grad);
        }
    }


    for (int i = 0; i < hiddenSize1; i++) {
        for (int j = 0; j < inputSize; j++) {
            float grad = layer1_output_gradient[i] * input[j];
            grad = fminf(fmaxf(grad, -1.0f), 1.0f);
            atomicAdd(&weightsLayer1[i * inputSize + j], -learningRate * grad);
        }
        atomicAdd(&biasLayer1[i], -learningRate * layer1_output_gradient[i]);
    }
}
