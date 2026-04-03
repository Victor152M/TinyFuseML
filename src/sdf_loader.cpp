#include <sdf_loader.h>

std::vector<SDFSample> LoadSDFSamples(const std::string& filename)
{
    std::vector<SDFSample> dataset;
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open SDF file: " << filename << std::endl;
        return dataset;
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t numSamples = fileSize / sizeof(SDFSample);
    dataset.resize(numSamples);

    file.read(reinterpret_cast<char*>(dataset.data()), fileSize);
    if (!file) {
        std::cerr << "Error reading SDF samples from file." << std::endl;
        dataset.clear();
    }

    file.close();
    std::cout << "Loaded " << dataset.size() << " SDF samples from " << filename << std::endl;
    return dataset;
}