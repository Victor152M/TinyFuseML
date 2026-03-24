#pragma once

struct ProgramArgs
{
    bool doSDFTraining = false;
    std:string imagePath;
    int trainingStepsMaximum = 1000;
    float batchLossThreshold = 0.0005f;
}

void PrintHelp();
ProgramArgs ParseArguments(int argc, char** argv):


