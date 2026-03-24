#pragma once
#include <vector>
#include <string>

bool LoadImage(const std::string& path, std::vector<float>& out_image, int& width, int& height);
