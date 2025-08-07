#pragma once

__global__ void mlpKernel(
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
    float* __restrict__ hashTable);