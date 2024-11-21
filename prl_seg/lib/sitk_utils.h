#pragma once

#include <string>
#include <filesystem>


namespace SimpleITKUtils
{

std::string applyN4BiasCorrection(const std::string &inputPath, const std::string &inputMaskPath, const std::string &outputPath, int shrinkFactor);

std::string addEchoDim(const std::string &inputPath, const std::string &outputPath);

}