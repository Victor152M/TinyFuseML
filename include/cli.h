#include <string>
#include <trainer.h>

struct CLIOptions
{
    ETrainingMode mode = ETrainingMode::Image;

    std::string imagePath;

    int trainingStepsMaximum = 2000;
    float batchLossThreshold = 0.005f;

    int renderW = 0;
    int renderH = 0;

    int baseHashResolution = 256;

    float learningRate = 0.0002f;
    float hashLearningRate = 0.4f;
};

CLIOptions ParseArguments(int argc, char** argv);
void PrintHelp();