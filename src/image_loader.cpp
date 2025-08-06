#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <image_loader.h>
#include <iostream>

bool loadImage(const std::string& path, std::vector<float>& out_image, int& width, int& height) {
    int channels;
    unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!img) {
        std::cerr << "Failed to load image: " << path << "\n";
        return false;
    }

    out_image.resize(width * height * 3);
    for (int i = 0; i < width * height * 3; ++i) {
        out_image[i] = img[i] / 255.0f;
    }

    stbi_image_free(img);
    return true;
}
