#include <iostream>
#include <chrono>
#include <trainer.h>
#include <hash_encoding.cuh>
#include <image_loader.h>
#include <cli.h>

int main(int argc, char** argv)
{
    CLIOptions options = ParseArguments(argc, argv);

    // Default training settings
    const int batchSize = 262144; // 2^18
    const int inputSize = HASH_ENCODED_SIZE;
    const int hiddenSize1 = 16;
    const int hiddenSize2 = 16;
    int outputSize = (options.mode == ETrainingMode::SDF) ? 1 : 3;

    std::vector<float> imageData;
    int imageWidth, imageHeight;

    CTrainer trainer(
        batchSize,
        inputSize,
        hiddenSize1,
        hiddenSize2,
        outputSize,
        options.learningRate,
        options.hashLearningRate,
        options.mode
    );

    if (options.mode == ETrainingMode::Image)
    {
        LoadImage(options.imagePath, imageData, imageWidth, imageHeight);
        trainer.SetImage(imageData, imageWidth, imageHeight);
    }

    std::cout << "Starting training ... \n";

    auto startTime = std::chrono::high_resolution_clock::now();

    trainer.Train(options.trainingStepsMaximum, options.batchLossThreshold, options.baseHashResolution);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    std::cout<< "Training completed in "
             << elapsed.count() << " seconds \n";
             
    trainer.Render(options.renderW, options.renderH, options.baseHashResolution);

    cudaDeviceReset();
    return 0;
}
