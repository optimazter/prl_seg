#pragma once

#include <string>
#include <format>
#include <cstdlib>
#include <filesystem>
#include <array>
#include <cmd.h>


class MEDI
{

public:
    MEDI(const std::string &matlabDir);
    std::string PDF(const std::string &inputT2PhasePath, const std::string &inputMaskPath, const std::string &outputPath, const std::array<double, 3> &voxelSize);

private:
    std::string matlabDir;
    std::string cmdPath;
    std::string pdfPath;

};