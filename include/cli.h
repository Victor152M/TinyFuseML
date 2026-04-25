#include <string>
#include <trainer.h>
#include <iostream>

struct CLIOptions
{
    ETrainingMode mode = ETrainingMode::Image;

    // default image path
    std::string imagePath = std::string(PROJECT_SOURCE_DIR) + "/images/cassette_shop_fullhd.png";
    // default sdf data path
    std::string sdfDataPath = std::string(PROJECT_SOURCE_DIR) + "/sdfs/greek_column.bin";

    int trainingStepsMaximum = 2000;
    float batchLossThreshold = 0.005f;

    int renderW = 0;
    int renderH = 0;

    int baseHashResolution = 256;

    float learningRate = 0.0002f;
    float hashLearningRate = 0.35f;
};

CLIOptions ParseArguments(int argc, char** argv);
void PrintHelp();