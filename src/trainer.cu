#include <trainer.h>
#include <volumetric_renderer.h>
#include <hash_encoding.cuh>
#include <ctime>
#include <iostream>
#include <mlp_kernel.cuh>
#include <opencv2/opencv.hpp>
#include <image_writer.h>

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
// Host Buffers
//
void CTrainer::InitHostBuffers()
{
    m_hWeightsLayer1 = new float[m_inputSize * m_hiddenSize1];
    m_hWeightsLayer2 = new float[m_hiddenSize1 * m_hiddenSize2];
    m_hWeightsLayer3 = new float[m_hiddenSize2 * m_outputSize];
    m_hBiasLayer1 = new float[m_hiddenSize1];
    m_hBiasLayer2 = new float[m_hiddenSize2];
    m_hBiasLayer3 = new float[m_outputSize];

    InitRandom(m_hWeightsLayer1, m_inputSize * m_hiddenSize1);
    InitRandom(m_hWeightsLayer2, m_hiddenSize1 * m_hiddenSize2);
    InitRandom(m_hWeightsLayer3, m_hiddenSize2 * m_outputSize);

    std::fill(m_hBiasLayer1, m_hBiasLayer1 + m_hiddenSize1, 0.0f);
    std::fill(m_hBiasLayer2, m_hBiasLayer2 + m_hiddenSize2, 0.0f);
    std::fill(m_hBiasLayer3, m_hBiasLayer3 + m_outputSize, 0.0f);

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

    cudaMemcpy(m_dWeightsLayer1, m_hWeightsLayer1, m_inputSize * m_hiddenSize1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_dBiasLayer1, m_hBiasLayer1, m_hiddenSize1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_dWeightsLayer2, m_hWeightsLayer2, m_hiddenSize1 * m_hiddenSize2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_dBiasLayer2, m_hBiasLayer2, m_hiddenSize2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_dWeightsLayer3, m_hWeightsLayer3, m_hiddenSize2 * m_outputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_dBiasLayer3, m_hBiasLayer3, m_outputSize * sizeof(float), cudaMemcpyHostToDevice);

    AllocateHashTable(&m_dHashTable);
}

// -----------------------------------------
// Free Host and Device Buffers
//
void CTrainer::FreeBuffers()
{
    // Host
    delete[] m_hWeightsLayer1;  m_hWeightsLayer1 = nullptr;
    delete[] m_hWeightsLayer2;  m_hWeightsLayer2 = nullptr;
    delete[] m_hWeightsLayer3;  m_hWeightsLayer3 = nullptr;
    delete[] m_hBiasLayer1;     m_hBiasLayer1 = nullptr;
    delete[] m_hBiasLayer2;     m_hBiasLayer2 = nullptr;
    delete[] m_hBiasLayer3;     m_hBiasLayer3 = nullptr;

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
    int threadsPerBlock = 120;
    int numBlocks = (m_batchSize + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * (m_hiddenSize1 + m_hiddenSize2 + m_outputSize + m_outputSize +
                                            m_hiddenSize2 + m_hiddenSize1) * sizeof(float);

    srand(time(0));

    for (int step = 0; step <= maxSteps; step++)
    {
        // Sample depending on mode
        SampleBatch(); // fills m_hPositions (and m_hTargets for image training)

        if (m_trainingMode == ETrainingMode::SDF)
        {
            cudaMemcpy(m_dPositions, m_hPositions.data(), m_batchSize * 3 * sizeof(float), cudaMemcpyHostToDevice);

            // Generate SDF targets on GPU
            GenerateSDFTargetsKernel<<<numBlocks, threadsPerBlock>>>(
                m_dPositions,
                m_dTargets,
                m_batchSize,
                0.0f
            );
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
                float diff = m_hOutput[i] - m_hTargets[i];
                batchLoss += diff * diff;
            }
            batchLoss /= m_batchSize;

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
        std::vector<unsigned char> outputImage(width * height * 4);
        unsigned char* d_outputRGBA;
        cudaMalloc(&d_outputRGBA, width * height * 4 * sizeof(unsigned char));

        dim3 threadsPerBlock2D(16, 16);
        dim3 numBlocks2D(
        (width + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
        (height + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y
        );

        float time = 0.0f;

        RenderKernel<<<numBlocks2D, threadsPerBlock2D>>>(
            d_outputRGBA, width, height, time,
            m_dWeightsLayer1, m_dBiasLayer1,
            m_dWeightsLayer2, m_dBiasLayer2,
            m_dWeightsLayer3, m_dBiasLayer3,
            m_dHashTable, baseHashResolution
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Render kernel error: %s\n", cudaGetErrorString(err));

        cudaMemcpy(outputImage.data(), d_outputRGBA, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        cv::Mat renderMat(height, width, CV_8UC4, outputImage.data());
        cv::cvtColor(renderMat, renderMat, cv::COLOR_RGBA2BGR); // Optional: convert to BGR
        cv::imwrite("sdf_render.png", renderMat);

        cudaFree(d_outputRGBA);
        return;
    }

    std::vector<unsigned char> outputImage(width * height * 3);

    const int inferenceBatchSize = 262144;
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

        int threadsPerBlock = 120;
        int numBlocks = (currentBatch + threadsPerBlock - 1) / threadsPerBlock;
        int sharedMemSize = threadsPerBlock * (m_hiddenSize1 + m_hiddenSize2 + m_outputSize + m_outputSize +
                                            m_hiddenSize2 + m_hiddenSize1) * sizeof(float);

        MLPKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
            m_dPositions, m_dWeightsLayer1, m_dBiasLayer1,
            m_dWeightsLayer2, m_dBiasLayer2,
            m_dWeightsLayer3, m_dBiasLayer3,
            m_dOutput, HASH_ENCODED_SIZE, m_hiddenSize1, m_hiddenSize2,
            m_outputSize, m_batchSize, m_learningRate, m_hashLearningRate, m_dTargets, false, m_dHashTable, baseHashResolution
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
void CTrainer::SampleBatch()
{
    if (m_trainingMode == ETrainingMode::SDF)
    {
        const float sphereRadius = 0.3f;

        for (int i = 0; i < m_batchSize; i++) 
        {
            float px, py, pz;
            float u = (float)rand() / RAND_MAX;

            if (u < 0.4f) {
                // exact surface samples
                float theta = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
                float phi   = acosf(2.0f * ((float)rand() / RAND_MAX) - 1.0f);
                float r = sphereRadius;

                px = r * sinf(phi) * cosf(theta);
                py = r * cosf(phi);
                pz = r * sinf(phi) * sinf(theta);
            }
            else if (u <= 0.7f) {
                // near surface
                float theta = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
                float phi   = acosf(2.0f * ((float)rand() / RAND_MAX) - 1.0f);

                float shellThickness = 0.1f;
                float offset = (((float)rand() / RAND_MAX) - 0.5f) * shellThickness;
                float r = sphereRadius + offset;

                px = r * sinf(phi) * cosf(theta);
                py = r * cosf(phi);
                pz = r * sinf(phi) * sinf(theta);
            }
            else {
                // uniform cube
                px = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                py = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                pz = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            }

            m_hPositions[i*3+0] = px;
            m_hPositions[i*3+1] = py;
            m_hPositions[i*3+2] = pz;
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
// Lunch Kernel
//
void CTrainer::LaunchKernel(int numBlocks, int threadsPerBlock, int baseHashResolution)
{
    int sharedMemSize = threadsPerBlock * (m_hiddenSize1 + m_hiddenSize2 + m_outputSize + m_outputSize + 
                        m_hiddenSize2 + m_hiddenSize1) * sizeof(float);

    if (m_trainingMode == ETrainingMode::SDF)
    {
        MLPSDFKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
            m_dPositions,
            m_dWeightsLayer1, m_dBiasLayer1,
            m_dWeightsLayer2, m_dBiasLayer2,
            m_dWeightsLayer3, m_dBiasLayer3,
            m_dOutput,
            m_inputSize, m_hiddenSize1, m_hiddenSize2, m_outputSize,
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

// -----------------------------------------
// Set image for image training
//
void CTrainer::SetImage(const std::vector<float>& data, int width, int height)
{
    m_imageData = data;
    m_imageWidth = width;
    m_imageHeight = height;
}
