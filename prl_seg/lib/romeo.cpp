#include "romeo.h"
#include <c_romeo.h>
#include <iostream>
#include <filesystem>
#include <vector>


ROMEO::ROMEO()
{
    JULIA_RUN();
}

std::string ROMEO::romeo(const std::string &inputPhasePath, const std::string &inputMaskPath, const std::string &outputPath)
{

    std::cout << "Running ROMEO\n";
    if (std::filesystem::exists(outputPath)) 
    {
        std::cout << "Output path " << outputPath << " already exists, returning it instead. \n";
        return outputPath;
    }

    c_romeo(inputPhasePath.c_str(), inputMaskPath.c_str(), outputPath.c_str());

    return outputPath;
}