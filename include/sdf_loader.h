#pragma once
#include <vector>
#include <iostream>
#include <fstream>

struct SDFSample { float x, y, z, sdf; };

std::vector<SDFSample> LoadSDFSamples(const std::string& filepath);