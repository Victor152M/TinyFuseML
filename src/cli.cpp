#include <cstdlib>
#include <cli.h>

void PrintHelp()
{
    std::cout << "TinyFuseML Usage:\n\n"
              << "--mode [image|sdf]\n"
              << "--image <path>\n"
              << "--sdfdata <path>\n"
              << "--steps <int>\n"
              << "--loss <float>\n"
              << "--res <width> <height>\n"
              << "--hashres <int>\n"
              << "--lr <float>\n"
              << "--hashlr <float>\n"
              << "--help\n\n";
}

CLIOptions ParseArguments(int argc, char** argv)
{
    CLIOptions options;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            PrintHelp();
            exit(0);
        }
        else if (arg == "--mode" && i + 1 < argc)
        {
            std::string mode = argv[++i];

            if (mode == "sdf")
            {
                options.mode = ETrainingMode::SDF;
                options.baseHashResolution = 3;
                options.batchLossThreshold = 0.000005;
                options.hashLearningRate = 0.05;
                options.renderW = 1920;
                options.renderH = 1080;
            }
            else
            {
                options.mode = ETrainingMode::Image;
            }
        }
        else if (arg == "--image" && i + 1 < argc)
        {
            options.imagePath = argv[++i];
        }
        else if (arg == "--steps" && i + 1 < argc)
        {
            options.trainingStepsMaximum = std::stoi(argv[++i]);
        }
        else if (arg == "--loss" && i + 1 < argc)
        {
            options.batchLossThreshold = std::stof(argv[++i]);
        }
        else if (arg == "--res" && i + 2 < argc)
        {
            options.renderW = std::stoi(argv[i + 1]);
            options.renderH = std::stoi(argv[i + 2]);
            i += 2;
        }
        else if (arg == "--hashres" && i + 1 < argc)
        {
            options.baseHashResolution = std::stoi(argv[++i]);
        }
        else if (arg == "--lr" && i + 1 < argc)
        {
            options.learningRate = std::stof(argv[++i]);
        }
        else if ((arg == "--sdfdata") && i + 1 < argc)
        {
            options.sdfDataPath = argv[++i];
        }
        else if (arg == "--hashlr" && i + 1 < argc)
        {
            options.hashLearningRate = std::stof(argv[++i]);
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            PrintHelp();
            exit(1);
        }
    }

    return options;
}