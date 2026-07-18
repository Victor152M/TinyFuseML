#include <trainer.h>
#include <volumetric_renderer.h>
#include <hash_encoding.cuh>
#include <ctime>
#include <iostream>
#include <mlp_kernel.cuh>
#include <opencv2/opencv.hpp>
#include <image_writer.h>

#ifndef PI
#define PI 3.14159265358f
#endif

// -----------------------------------------
// Constructor
//
CTrainer::CTrainer(int batchSize, int inputSize, int hiddenSize1, int hiddenSize2, int outputSize,
                         float learningRate, float hashLearningRate, ETrainingMode trainingMode)
    :   m_batchSize(batchSize), m_inputSize(inputSize),
        m_hiddenSize1(hiddenSize1), m_hiddenSize2(hiddenSize2), m_outputSize(outputSize),
        m_learningRate(learningRate), m_hashLearningRate(hashLearningRate), m_trainingMode(trainingMode) 
{
    InitHostBuffers();
    InitDeviceBuffers();
}

// -----------------------------------------
// Destructor
//
CTrainer::~CTrainer()
{
    FreeBuffers();
}

// -----------------------------------------
// Random Initialization Helper
//
void CTrainer::InitRandom(float* array, int size, float scale)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = static_cast<float>(rand()) / RAND_MAX * scale;
    }
}

// -----------------------------------------
// Setters
//
void CTrainer::SetSDFDataset(const std::vector<SDFSample>& dataset)
{
    m_sdfDataset = dataset;
}

void CTrainer::SetImage(const std::vector<float>& data, int width, int height)
{
    m_imageData = data;
    m_imageWidth = width;
    m_imageHeight = height;
}

// -----------------------------------------
// Host Buffers
//
void CTrainer::InitHostBuffers()
{
    m_hWeightsLayer1.resize(m_inputSize * m_hiddenSize1);
    m_hWeightsLayer2.resize(m_hiddenSize1 * m_hiddenSize2);
    m_hWeightsLayer3.resize(m_hiddenSize2 * m_outputSize);

    m_hBiasLayer1.resize(m_hiddenSize1);
    m_hBiasLayer2.resize(m_hiddenSize2);
    m_hBiasLayer3.resize(m_outputSize);

    InitRandom(m_hWeightsLayer1.data(), m_hWeightsLayer1.size());
    InitRandom(m_hWeightsLayer2.data(), m_hWeightsLayer2.size());
    InitRandom(m_hWeightsLayer3.data(), m_hWeightsLayer3.size());

    std::fill(m_hBiasLayer1.begin(), m_hBiasLayer1.end(), 0.0f);
    std::fill(m_hBiasLayer2.begin(), m_hBiasLayer2.end(), 0.0f);
    std::fill(m_hBiasLayer3.begin(), m_hBiasLayer3.end(), 0.0f);

    m_hPositions.resize(m_batchSize * 3);
    m_hTargets.resize(m_batchSize * m_outputSize);
    m_hOutput.resize(m_batchSize * m_outputSize);
}

// -----------------------------------------
// Device Buffers
//
void CTrainer::InitDeviceBuffers()
{
    // Determine sizes depending on training mode
    int posSize = (m_trainingMode == ETrainingMode::SDF) ? 3 : 2;
    int targetSize = (m_trainingMode == ETrainingMode::SDF) ? 1 : m_outputSize;

    cudaMalloc(&m_dPositions, m_batchSize * posSize * sizeof(float));
    cudaMalloc(&m_dTargets, m_batchSize * targetSize * sizeof(float));
    cudaMalloc(&m_dWeightsLayer1, m_inputSize * m_hiddenSize1 * sizeof(float));
    cudaMalloc(&m_dBiasLayer1, m_hiddenSize1 * sizeof(float));
    cudaMalloc(&m_dWeightsLayer2, m_hiddenSize1 * m_hiddenSize2 * sizeof(float));
    cudaMalloc(&m_dBiasLayer2, m_hiddenSize2 * sizeof(float));
    cudaMalloc(&m_dWeightsLayer3, m_hiddenSize2 * m_outputSize * sizeof(float));
    cudaMalloc(&m_dBiasLayer3, m_outputSize * sizeof(float));
    cudaMalloc(&m_dOutput, m_batchSize * m_outputSize * sizeof(float));

    cudaMemcpy(m_dWeightsLayer1, m_hWeightsLayer1.data(), m_inputSize * m_hiddenSize1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_dBiasLayer1, m_hBiasLayer1.data(), m_hiddenSize1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_dWeightsLayer2, m_hWeightsLayer2.data(), m_hiddenSize1 * m_hiddenSize2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_dBiasLayer2, m_hBiasLayer2.data(), m_hiddenSize2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_dWeightsLayer3, m_hWeightsLayer3.data(), m_hiddenSize2 * m_outputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_dBiasLayer3, m_hBiasLayer3.data(), m_outputSize * sizeof(float), cudaMemcpyHostToDevice);

    AllocateHashTable(&m_dHashTable);
}

// -----------------------------------------
// Free Host and Device Buffers
//
void CTrainer::FreeBuffers()
{
    // Device
    if (m_dPositions)  cudaFree(m_dPositions); m_dPositions = nullptr;
    if (m_dTargets)    cudaFree(m_dTargets); m_dTargets = nullptr;
    if (m_dWeightsLayer1) cudaFree(m_dWeightsLayer1); m_dWeightsLayer1 = nullptr;
    if (m_dWeightsLayer2) cudaFree(m_dWeightsLayer2); m_dWeightsLayer2 = nullptr;
    if (m_dWeightsLayer3) cudaFree(m_dWeightsLayer3); m_dWeightsLayer3 = nullptr;
    if (m_dBiasLayer1)    cudaFree(m_dBiasLayer1); m_dBiasLayer1 = nullptr;
    if (m_dBiasLayer2)    cudaFree(m_dBiasLayer2); m_dBiasLayer2 = nullptr;
    if (m_dBiasLayer3)    cudaFree(m_dBiasLayer3); m_dBiasLayer3 = nullptr;
    if (m_dOutput)        cudaFree(m_dOutput); m_dOutput = nullptr;
    if (m_dHashTable)     cudaFree(m_dHashTable); m_dHashTable = nullptr;
}

// -----------------------------------------
// Train
//
void CTrainer::Train(int maxSteps, float batchLossThreshold, int baseHashResolution)
{
    int threadsPerBlock = 128;
    int numBlocks = (m_batchSize + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * (m_hiddenSize1 + m_hiddenSize2 + m_outputSize + m_outputSize +
                                            m_hiddenSize2 + m_hiddenSize1) * sizeof(float);

    srand(time(0));

    for (int step = 0; step <= maxSteps; step++)
    {
        // Sample depending on mode
        SampleBatch(step); // fills m_hPositions (and m_hTargets for image training)

        if (m_trainingMode == ETrainingMode::SDF)
        {
            cudaMemcpy(m_dPositions, m_hPositions.data(), m_batchSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(m_dTargets, m_hTargets.data(), m_batchSize * sizeof(float), cudaMemcpyHostToDevice);
        }
        else if (m_trainingMode == ETrainingMode::Image)
        {
            cudaMemcpy(m_dPositions, m_hPositions.data(), m_batchSize * 2 *sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(m_dTargets, m_hTargets.data(), m_batchSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
        }

        LaunchKernel(numBlocks, threadsPerBlock, baseHashResolution);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        // Logging and loss computation
        if (step % 100 == 0)
        {
            // Copy targets and outputs back for loss computation
            cudaMemcpy(m_hTargets.data(), m_dTargets, m_batchSize * m_outputSize * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(m_hOutput.data(), m_dOutput, m_batchSize * m_outputSize * sizeof(float), cudaMemcpyDeviceToHost);

            float batchLoss = 0.0f;
            for (int i = 0; i < m_batchSize; i++)
            {
                for (int c = 0; c < m_outputSize; c++)
                {
                    int idx = i * m_outputSize + c;
                    float diff = m_hOutput[idx] - m_hTargets[idx];
                    batchLoss += diff * diff;
                }
            }

            batchLoss /= (m_batchSize * m_outputSize);

            std::cout << " Step " << step << ", Loss: " << batchLoss << std::endl;

            if (m_trainingMode == ETrainingMode::SDF)
            {
                std::cout << "Sample position: (" 
                << m_hPositions[0*3+0] << ", "
                << m_hPositions[0*3+1] << ", "
                << m_hPositions[0*3+2] << ") "
                << "Target: " << m_hTargets[0] 
                << " Output: " << m_hOutput[0] << std::endl;
            }
            
            if (batchLoss < batchLossThreshold)
                break;
        }
    }
};

// -----------------------------------------
// Render the output
//
void CTrainer::Render(int width, int height, int baseHashResolution)
{
    if (m_trainingMode == ETrainingMode::SDF)
    {
        float camAngles[3] = {PI/2, PI, 5*PI/3 };

        for (int view = 0; view < 3; ++view)
        {
            float time = camAngles[view];
            std::vector<unsigned char> outputImage(width * height * 4);
            unsigned char* d_outputRGBA;
            cudaMalloc(&d_outputRGBA, width * height * 4 * sizeof(unsigned char));

            dim3 threadsPerBlock2D(16, 16);
            dim3 numBlocks2D(
                (width + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
                (height + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y
            );

            RenderKernel<<<numBlocks2D, threadsPerBlock2D>>>(
                d_outputRGBA, width, height, time, // time now acts as camera rotation angle
                m_dWeightsLayer1, m_dBiasLayer1,
                m_dWeightsLayer2, m_dBiasLayer2,
                m_dWeightsLayer3, m_dBiasLayer3,
                m_dHashTable, baseHashResolution
            );

            cudaDeviceSynchronize();
            cudaMemcpy(outputImage.data(), d_outputRGBA, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

            cv::Mat renderMat(height, width, CV_8UC4, outputImage.data());
            cv::cvtColor(renderMat, renderMat, cv::COLOR_RGBA2BGR);
            
            std::string filename = "sdf_render_view" + std::to_string(view) + ".png";
            cv::imwrite(filename, renderMat);

            cudaFree(d_outputRGBA);
        }
        return;
    }

    std::vector<unsigned char> outputImage(width * height * 3);

    const int inferenceBatchSize = m_batchSize;
    std::vector<float> h_batchPositions(inferenceBatchSize * 2);
    std::vector<float> h_batchOutput(inferenceBatchSize * 3);

    for (int start = 0; start < width * height; start += inferenceBatchSize) 
    {
        int currentBatch = std::min(inferenceBatchSize, width * height - start);
        for (int i = 0; i < currentBatch; ++i) 
        {
            int idx = start + i;
            int px = idx % width;
            int py = idx / width;
            float x = px / float(width);
            float y = py / float(height);
            h_batchPositions[i * 2 + 0] = x;
            h_batchPositions[i * 2 + 1] = y;
        }

        cudaMemcpy(m_dPositions, h_batchPositions.data(), currentBatch * 2 * sizeof(float), cudaMemcpyHostToDevice);

        int threadsPerBlock = 128;
        int numBlocks = (currentBatch + threadsPerBlock - 1) / threadsPerBlock;
        int sharedMemSize = threadsPerBlock * (m_hiddenSize1 + m_hiddenSize2 + m_outputSize + m_outputSize +
                                            m_hiddenSize2 + m_hiddenSize1) * sizeof(float);

        MLPKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
            m_dPositions, m_dWeightsLayer1, m_dBiasLayer1,
            m_dWeightsLayer2, m_dBiasLayer2,
            m_dWeightsLayer3, m_dBiasLayer3,
            m_dOutput, HASH_ENCODED_SIZE, m_hiddenSize1, m_hiddenSize2,
            m_outputSize, m_batchSize, m_learningRate, m_hashLearningRate, nullptr, false, m_dHashTable, baseHashResolution
        );

        cudaMemcpy(h_batchOutput.data(), m_dOutput, currentBatch * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < currentBatch; ++i) 
        {
            int idx = start + i;
            int rgbIdx = idx * 3;

            float r = h_batchOutput[i * 3 + 0];
            float g = h_batchOutput[i * 3 + 1];
            float b = h_batchOutput[i * 3 + 2];

            outputImage[rgbIdx + 0] = (unsigned char)(fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f);
            outputImage[rgbIdx + 1] = (unsigned char)(fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f);
            outputImage[rgbIdx + 2] = (unsigned char)(fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f);
        }
    }

    SaveRgbImage(outputImage.data(), width, height, "Full_Inference.png");
};

// -----------------------------------------
// Sample Batch
//
void CTrainer::SampleBatch(int step)
{
    if (m_trainingMode == ETrainingMode::SDF)
    {
        for (int i = 0; i < m_batchSize; i++)
        {
            int idx = (step * m_batchSize + i) % m_sdfDataset.size();
            m_hPositions[i*3 + 0] = m_sdfDataset[idx].x;
            m_hPositions[i*3 + 1] = m_sdfDataset[idx].y;
            m_hPositions[i*3 + 2] = m_sdfDataset[idx].z;

            m_hTargets[i] = m_sdfDataset[idx].sdf;
        }
    } else
    {
        for (int i = 0; i < m_batchSize; i++) 
        {
            int px = rand() % m_imageWidth;
            int py = rand() % m_imageHeight;

            // Normalize x, y to [0, 1]
            float x = px / float(m_imageWidth);
            float y = py / float(m_imageHeight);

            m_hPositions[i * 2 + 0] = x;
            m_hPositions[i * 2 + 1] = y;

            // Get RGB target
            int idx = (py * m_imageWidth + px) * 3;
            m_hTargets[i * m_outputSize + 0] = m_imageData[idx + 0];
            m_hTargets[i * m_outputSize + 1] = m_imageData[idx + 1];
            m_hTargets[i * m_outputSize + 2] = m_imageData[idx + 2];
        };
    }
}

// -----------------------------------------
// Sample SDF Batch From File 
//
void CTrainer::SampleSDFBatchFromFile(const std::vector<SDFSample>& dataset)
{
    for (int i = 0; i < m_batchSize; i++)
    {
        int idx = rand() % dataset.size();
        m_hPositions[i*3 + 0] = dataset[idx].x;
        m_hPositions[i*3 + 1] = dataset[idx].y;
        m_hPositions[i*3 + 2] = dataset[idx].z;

        m_hTargets[i] = dataset[idx].sdf;
    }
}

// -----------------------------------------
// Lunch Kernel
//
void CTrainer::LaunchKernel(int numBlocks, int threadsPerBlock, int baseHashResolution)
{
    int sharedMemSize = (m_outputSize * m_hiddenSize2 + m_outputSize + m_hiddenSize2 * m_hiddenSize1 + m_hiddenSize2 +
                        m_hiddenSize1 * m_inputSize + m_hiddenSize1) * sizeof(float);

    if (m_trainingMode == ETrainingMode::SDF)
    {
        MLPSDFKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
            m_dPositions,
            m_dWeightsLayer1, m_dBiasLayer1,
            m_dWeightsLayer2, m_dBiasLayer2,
            m_dWeightsLayer3, m_dBiasLayer3,
            m_dOutput,
            m_inputSize, m_hiddenSize1,m_hiddenSize2, m_outputSize,
            m_batchSize,
            m_learningRate,
            m_hashLearningRate,
            m_dTargets,
            true,
            m_dHashTable,
            baseHashResolution
        );
    } else 
    {
        MLPKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
                m_dPositions, m_dWeightsLayer1, m_dBiasLayer1,
                m_dWeightsLayer2, m_dBiasLayer2,
                m_dWeightsLayer3, m_dBiasLayer3,
                m_dOutput, HASH_ENCODED_SIZE, m_hiddenSize1, m_hiddenSize2,
                m_outputSize, m_batchSize, m_learningRate, m_hashLearningRate, m_dTargets, true, m_dHashTable, baseHashResolution);
    }
}
