#include <cassert>
#include <cfloat>
#include <iostream>
#include <cuda_runtime.h>
#include <mlp_kernel.cuh>
#include <ostream>
#include <image_loader.h>
#include <image_writer.h>
#include <cstdlib> // for rand()
#include <ctime> // for seeding
#include <fourier_encoding.h>
#include <hash_encoding.cuh>
#include <opencv2/opencv.hpp>
#include <hash_encoding.cuh>
#include <chrono>



float* d_hashTable;

int main() {
    std::vector<float> img_data;
    int width, height;

    loadImage("../training_data/mountain.png", img_data, width, height);
    std::cout << "Image loaded: " << width << "x" << height << std::endl;

    cv::Mat preview(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    allocateHashTable(&d_hashTable);

    const int inputSize = 2;
    const int hiddenSize1 = 16;
    const int hiddenSize2 = 16;
    const int outputSize = 3;
    const int batchSize = 524288; //2 ^ 18 = 262144

    std::vector<float> h_positions(batchSize * 2);
    std::vector<float> h_output(batchSize * outputSize);
    
    float h_weightsLayer1[inputSize * hiddenSize1];
    float h_biasLayer1[hiddenSize1];

    for (int i = 0; i < inputSize * hiddenSize1; i++) {
        h_weightsLayer1[i] = static_cast<float>(rand()) / RAND_MAX * 0.3f;
    }
    for (int i = 0; i < hiddenSize1; i++) {
        h_biasLayer1[i] = 0.0f;
    }

    float h_weightsLayer2[hiddenSize1 * hiddenSize2];
    float h_biasLayer2[hiddenSize2];

    for (int i = 0; i < hiddenSize1 * hiddenSize2; i++) {
        h_weightsLayer2[i] = static_cast<float>(rand()) / RAND_MAX * 0.3f;
    }

    for (int i = 0; i < hiddenSize2; i++) {
        h_biasLayer2[i] = 0.0f;
    }

    float h_weightsLayer3[hiddenSize2 * outputSize];
    float h_biasLayer3[outputSize];

    for (int i = 0; i < hiddenSize2 * outputSize; i++) {
    h_weightsLayer3[i] = static_cast<float>(rand()) / RAND_MAX * 0.3f;
    }

    for (int i = 0; i < outputSize; i++) {
        h_biasLayer3[i] = 0.0f;
    }

    float h_target[batchSize * outputSize];

    float learningRate = 0.0005f;

    float *d_weightsLayer1, *d_weightsLayer2, *d_biasLayer1, *d_biasLayer2, *d_output, *d_target;
    float *d_weightsLayer3, *d_biasLayer3, *d_positions = nullptr;

    cudaMalloc(&d_positions, batchSize * 2 * sizeof(float));
    cudaMalloc(&d_weightsLayer1, inputSize * hiddenSize1 * sizeof(float));
    cudaMalloc(&d_weightsLayer2, hiddenSize1 * hiddenSize2 * sizeof(float));
    cudaMalloc(&d_biasLayer1, hiddenSize1 * sizeof(float));
    cudaMalloc(&d_biasLayer2, hiddenSize2 * sizeof(float));
    cudaMalloc(&d_output, batchSize * outputSize * sizeof(float));
    cudaMalloc(&d_target,        batchSize * outputSize * sizeof(float));
    cudaMalloc(&d_weightsLayer3, hiddenSize2 * outputSize * sizeof(float));
    cudaMalloc(&d_biasLayer3, outputSize * sizeof(float));

    cudaMemcpy(d_weightsLayer1, h_weightsLayer1, inputSize * hiddenSize1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsLayer2, h_weightsLayer2, hiddenSize1 * hiddenSize2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biasLayer1, h_biasLayer1, hiddenSize1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biasLayer2, h_biasLayer2, hiddenSize2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biasLayer3, h_biasLayer3,  outputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsLayer3, h_weightsLayer3,  hiddenSize2 * outputSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;

    int threadsPerBlock = 160;
    int blocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    srand(time(0)); // seed RNG once

    int sharedMemSize = threadsPerBlock * (hiddenSize1 + hiddenSize2 + outputSize + outputSize + hiddenSize2 + hiddenSize1) * sizeof(float);

    std::cout << "Launching kernel with shared_mem_size = " << sharedMemSize << " bytes\n";

    std::cout << "Launching with: "
        << "blocks=" << blocks << ", "
        << "threadsPerBlock=" << threadsPerBlock << ", "
        << "shared_mem_size=" << sharedMemSize << ", "
        << "batch_size=" << batchSize << ", "
        << "input_size=" << inputSize << ", "
        << "hidden_size1=" << hiddenSize1 << ", "
        << "hidden_si2e=" << hiddenSize2 << ", "
        << "output_size=" << outputSize << std::endl;

    std::vector<unsigned char> output_image(width * height * 3);


    auto startTime = std::chrono::high_resolution_clock::now();

    for (int step = 0; step <= 2000; step++) {
        for (int i = 0; i < batchSize; i++) {
            int px = rand() % width;
            int py = rand() % height;

            // Normalize x, y to [0, 1]
            float x = px / float(width);
            float y = py / float(height);

            h_positions[i * 2 + 0] = x;
            h_positions[i * 2 + 1] = y;

            // Get RGB target
            int idx = (py * width + px) * 3;
            h_target[i * outputSize + 0] = img_data[idx + 0];
            h_target[i * outputSize + 1] = img_data[idx + 1];
            h_target[i * outputSize + 2] = img_data[idx + 2];
        };


        cudaMemcpy(d_positions, h_positions.data(), batchSize*2*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_target,    h_target,          batchSize*3*sizeof(float), cudaMemcpyHostToDevice);

        mlpKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
            d_positions, d_weightsLayer1, d_biasLayer1,
            d_weightsLayer2, d_biasLayer2,
            d_weightsLayer3, d_biasLayer3,
            d_output, HASH_ENCODED_SIZE, hiddenSize1, hiddenSize2,
            outputSize, batchSize, learningRate, d_target, true, d_hashTable);

        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        // Get a random pixel from batch
        int random_thread = rand() % batchSize;
        assert(random_thread >= 0 && random_thread < batchSize);
        cudaMemcpy(h_output.data(), d_output, batchSize * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        
        for (int i = 0; i < batchSize; ++i) {
            float jitter_x = ((rand() % 100) / 100.0f - 0.5f) / width;  // ~ ±0.5 px jitter
            float jitter_y = ((rand() % 100) / 100.0f - 0.5f) / height;

            float fx = std::min(std::max(h_positions[i * 2 + 0] + jitter_x, 0.0f), 1.0f);
            float fy = std::min(std::max(h_positions[i * 2 + 1] + jitter_y, 0.0f), 1.0f);


            int px = std::min(static_cast<int>(fx * width), width - 1);
            int py = std::min(static_cast<int>(fy * height), height - 1);

            if (px >= 0 && px < width && py >= 0 && py < height) {
                float* rgb = &h_output[i * 3];
                cv::Vec3b& pixel = preview.at<cv::Vec3b>(py, px);
                pixel[2] = static_cast<unsigned char>(fminf(fmaxf(rgb[0], 0.0f), 1.0f) * 255.0f); // R
                pixel[1] = static_cast<unsigned char>(fminf(fmaxf(rgb[1], 0.0f), 1.0f) * 255.0f); // G
                pixel[0] = static_cast<unsigned char>(fminf(fmaxf(rgb[2], 0.0f), 1.0f) * 255.0f); // B
            }
        }

        if (step % 1 == 0) {
            cv::Mat display;
            cv::resize(preview, display, cv::Size(), 1, 1, cv::INTER_NEAREST);
            cv::imshow("Live Training", display);
            cv::waitKey(1);
        }

        if (step % 1 == 0) {
            //learningRate *= 0.98;
            float batchLoss = 0.0f;
            for (int i = 0; i < batchSize * 3; ++i) {
                float diff = h_output[i] - h_target[i];
                batchLoss += diff * diff;
            }
            batchLoss /= (batchSize * 3); // Mean squared error

            std::cout << "Step " << step << ", Loss: " << batchLoss << std::endl;


            if (batchLoss < 0.0005f){
                auto endTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> totalElapsed = endTime - startTime;
                std::cout << "Training completed in " << totalElapsed.count() << " seconds.\n";
                break;
            }

            learningRate = std::max(learningRate, 1e-4f);
        }

        if (step % 25 == 0){
            learningRate *= 0.9;
        }
    }


    // Inference

    const int inferenceBatchSize = 8192; // or tune this depending on your GPU
    std::vector<float> h_batchPositions(inferenceBatchSize * 2);
    std::vector<float> h_batchOutput(inferenceBatchSize * 3);

    for (int start = 0; start < width * height; start += inferenceBatchSize) {
        int currentBatch = std::min(inferenceBatchSize, width * height - start);

        for (int i = 0; i < currentBatch; ++i) {
            int idx = start + i;
            int px = idx % width;
            int py = idx / width;
            float x = px / float(width);
            float y = py / float(height);
            h_batchPositions[i * 2 + 0] = x;
            h_batchPositions[i * 2 + 1] = y;
        }

        cudaMemcpy(d_positions, h_batchPositions.data(), currentBatch * 2 * sizeof(float), cudaMemcpyHostToDevice);

        mlpKernel<<<(currentBatch + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, sharedMemSize>>>(
            d_positions, d_weightsLayer1, d_biasLayer1,
            d_weightsLayer2, d_biasLayer2,
            d_weightsLayer3, d_biasLayer3,
            d_output, HASH_ENCODED_SIZE, hiddenSize1, hiddenSize2,
            outputSize, currentBatch, learningRate, d_target,  // dummy
            false, d_hashTable
        );

        cudaDeviceSynchronize();

        cudaMemcpy(h_batchOutput.data(), d_output, currentBatch * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < currentBatch; ++i) {
            int idx = start + i;
            int rgbIdx = idx * 3;

            float r = h_batchOutput[i * 3 + 0];
            float g = h_batchOutput[i * 3 + 1];
            float b = h_batchOutput[i * 3 + 2];

            output_image[rgbIdx + 0] = (unsigned char)(fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f);
            output_image[rgbIdx + 1] = (unsigned char)(fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f);
            output_image[rgbIdx + 2] = (unsigned char)(fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f);
        }
    }

    saveRgbImage(output_image.data(), width, height, "Full_Inference.png");

    cudaMemcpy(h_output.data(), d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < outputSize; i++) {
        std::cout << "Neuron " << i << ": " << h_output[i] << std::endl;
    }

    // Cleanup
    cudaFree(d_positions);
    cudaFree(d_weightsLayer1);
    cudaFree(d_biasLayer1);
    cudaFree(d_weightsLayer2);
    cudaFree(d_biasLayer2);
    cudaFree(d_output);
    cudaFree(d_hashTable);
}



