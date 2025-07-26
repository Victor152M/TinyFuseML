#pragma once
#include <vector>
#include <string>

bool load_image(const std::string& path, std::vector<float>& out_image, int& width, int& height);
