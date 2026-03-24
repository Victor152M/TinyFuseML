#pragma once
#include <vector>


enum class ETrainingMode
{
    SDF,
    Image
};

class CTrainer
{
public:
    CTrainer(int batchSize, int inputSize, int hiddenSize1, int hiddenSize2, int outputSize, float learningRate, 
        float hashLearningRate, ETrainingMode trainingMode);

    ~CTrainer();

    void InitHostBuffers();
    void InitDeviceBuffers();
    void FreeBuffers();

    void Train(int maxSteps, float batchLossThreshold, int baseHashResolution);
    void Render(int width, int height, int baseHashResolution);

    void SetImage(const std::vector<float>& data, int width, int height);

private:
    ETrainingMode m_trainingMode;

    // -----------------------------------------
	// Image data (if Image mode)
    std::vector<float> m_imageData;
    int m_imageWidth;
    int m_imageHeight;

    // -----------------------------------------
	// Host Buffers
    // 
    std::vector<float> m_hPositions;
    std::vector<float> m_hTargets;
    std::vector<float> m_hOutput;

    float* m_hWeightsLayer1 = nullptr;
    float* m_hWeightsLayer2 = nullptr;
    float* m_hWeightsLayer3 = nullptr;
    float* m_hBiasLayer1 = nullptr;
    float* m_hBiasLayer2 = nullptr;
    float* m_hBiasLayer3 = nullptr;

    // -----------------------------------------
	// Device Buffers
    //
    float* m_dPositions = nullptr;
    float* m_dTargets = nullptr;
    float* m_dWeightsLayer1 = nullptr;
    float* m_dWeightsLayer2 = nullptr;
    float* m_dWeightsLayer3 = nullptr;
    float* m_dBiasLayer1 = nullptr;
    float* m_dBiasLayer2 = nullptr;
    float* m_dBiasLayer3 = nullptr;
    float* m_dOutput = nullptr;
    float* m_dHashTable = nullptr;

    // -----------------------------------------
	// Network Parameters
    //
    int m_batchSize;
    int m_inputSize;
    int m_hiddenSize1;
    int m_hiddenSize2;
    int m_outputSize;
    float m_learningRate;
    float m_hashLearningRate;

    void SampleBatch();
    void LaunchKernel(int numBlocks, int threadsPerBlock, int baseHashResolution);

    // -----------------------------------------
	// Helpers
    //
    void InitRandom(float* array, int size, float scale = 0.3f);
};