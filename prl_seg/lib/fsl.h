#pragma once

#include <format>
#include <string>
#include <vector>
#include <filesystem>
#include <cmd.h>
#include <stdlib.h>
#include <boost/algorithm/string/join.hpp>

class FSL
{

public:

    FSL(const std::string &fslDir);


    std::string BET(const std::string &inputPath, const std::string &outputPath);
    std::string merge(std::vector<std::string> inputPaths, const std::string &outputPath);



private:

    std::string cmdPath;
    std::string fslDir;

};

