#include <iostream>
#include <cuda_runtime.h>
#include <mlp_kernel.cuh>
#include <ostream>
#include <image_loader.h>
#include <image_writer.h>
#include <cstdlib>  // for rand()
#include <ctime>    // for seeding

int main() {
    std::vector<float> img_data;
    int width, height;

    load_image("../training_data/dipper.png", img_data, width, height);
    std::cout << "Image loaded: " << width << "x" << height << std::endl;

    const int input_size = 3;
    const int output_size = 3;
    const int hidden_size = 3;

    float h_input[input_size] = {0.23, 0.32, 0.93};
    float h_output[output_size] = {};
    
    float h_weightsLayer1[] = {
        0.5f, 0.3f, 0.8f,   // Neuron 0
        0.2f, 0.4f, 0.1f,   // Neuron 1
        0.3f, 0.5f, 0.2f    // Neuron 2
    };
    float h_biasLayer1[3] = {0.1f, 0.2f, 0.3f};

    float h_weightsLayer2[] = {
        0.05f, 0.3f, 0.1f,   // Neuron 0
        0.2f, 0.9f, 0.1f,   // Neuron 1
        0.4f, 0.5f, 0.2f    // Neuron 2
    };
    float h_biasLayer2[3] = {0.1f, 0.2f, 0.3f};

    float h_target[output_size] = {1, 1, 1};

    float learning_rate = 0.01;

    float *d_input, *d_weightsLayer1, *d_weightsLayer2, *d_biasLayer1, *d_biasLayer2, *d_output, *d_target;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_weightsLayer1, 9 * sizeof(float));
    cudaMalloc(&d_weightsLayer2, 9 * sizeof(float));
    cudaMalloc(&d_biasLayer1, input_size * sizeof(float));
    cudaMalloc(&d_biasLayer2, input_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));
    cudaMalloc(&d_target,        output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsLayer1, h_weightsLayer1, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsLayer2, h_weightsLayer2, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biasLayer1, h_biasLayer1, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biasLayer2, h_biasLayer2, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target,        h_target,        output_size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 3;
    int blocks = 1;

    srand(time(0)); // seed RNG once

    for (int step = 0; step < 10000; step++) {
        int px = rand() % width;
        int py = rand() % height;

        // Normalize x, y to [0, 1]
        float x = px / float(width);
        float y = py / float(height);

        float h_input[3] = {x, y, 1.0f};

        // Get RGB target
        int idx = (py * width + px) * 3;
        float target[3] = {
            img_data[idx + 0],
            img_data[idx + 1],
            img_data[idx + 2]
        };

        cudaMemcpy(d_input, h_input, 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_target, target, 3 * sizeof(float), cudaMemcpyHostToDevice);

        bool training = true;

        mlp_kernel<<<1, 3>>>(
            d_input, d_weightsLayer1, d_biasLayer1,
            d_weightsLayer2, d_biasLayer2,
            d_output, input_size, output_size, 
            hidden_size, learning_rate, d_target, true);

        cudaDeviceSynchronize();

        cudaMemcpy(h_output, d_output, 3 * sizeof(float), cudaMemcpyDeviceToHost);

        if (step % 1000 == 0) {
            float loss = 0.0f;
            for (int i = 0; i < 3; ++i)
                loss += (h_output[i] - target[i]) * (h_output[i] - target[i]);

            std::cout << "Step " << step << ", Loss: " << loss << std::endl;
            std::cout << "Pixel (" << px << ", " << py << "): RGB = "
                    << h_output[0] << " "
                    << h_output[1] << " "
                    << h_output[2] << std::endl;
        }
    }

    std::vector<unsigned char> output_image(width * height * 3);

    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {
            float x = px / float(width);
            float y = py / float(height);
            float h_input[3] = {x, y, 1.0f};

            cudaMemcpy(d_input, h_input, 3 * sizeof(float), cudaMemcpyHostToDevice);

            mlp_kernel<<<1, 3>>>(
                d_input, d_weightsLayer1, d_biasLayer1,
                d_weightsLayer2, d_biasLayer2,
                d_output, input_size, hidden_size,
                output_size, learning_rate, d_target,  // unused dummy
                false
            );
            cudaDeviceSynchronize();

            cudaMemcpy(h_output, d_output, 3 * sizeof(float), cudaMemcpyDeviceToHost);

            // Clamp and store
            int idx = (py * width + px) * 3;
            output_image[idx + 0] = (unsigned char)(fminf(fmaxf(h_output[0], 0.0f), 1.0f) * 255.0f);
            output_image[idx + 1] = (unsigned char)(fminf(fmaxf(h_output[1], 0.0f), 1.0f) * 255.0f);
            output_image[idx + 2] = (unsigned char)(fminf(fmaxf(h_output[2], 0.0f), 1.0f) * 255.0f);
        }
    }

    save_rgb_image(output_image.data(), width, height, "reconstructed.png");

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < output_size; i++) {
        std::cout << "Neuron " << i << ": " << h_output[i] << std::endl;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weightsLayer1);
    cudaFree(d_biasLayer1);
    cudaFree(d_weightsLayer2);
    cudaFree(d_biasLayer2);
    cudaFree(d_output);
}



