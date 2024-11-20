#pragma once

#include <string>
#include <format>
#include <cstdlib>
#include <filesystem>
#include <cmd.h>


class MEDI
{

public:
    MEDI(const std::string &matlabDir);
    std::string PDF(const std::string &inputT2PhasePath, const std::string &inputBETPath, const std::string &outputPath);

private:
    std::string matlabDir;
    std::string cmdPath;
    std::string pdfPath;

};