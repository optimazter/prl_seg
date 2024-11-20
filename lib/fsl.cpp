#include "fsl.h"
#include <filesystem>

FSL::FSL(const std::string &fslDir)
{
    cmdPath = (std::filesystem::current_path() / "libs/FSL/fsl_cmd.sh").string();
    this->fslDir = fslDir;
    std::string chmodCmd = std::format("chmod u+x {}", cmdPath);
    system(chmodCmd.c_str());
}


std::string FSL::BET(const std::string &inputPath, const std::string &outputPath)
{
    std::string betCmd = std::format(
        "bash {} -f {} -c 'bet {} {}'", 
        cmdPath,
        fslDir,
        inputPath,
        outputPath
    );
    return runCommand("FSL BET", outputPath, betCmd);
};


std::string FSL::merge(std::vector<std::string> inputPaths, const std::string &outputPath)
{
    std::string inputStrings = boost::algorithm::join(inputPaths, " ");
    std::string mergeCmd = std::format(
        "bash {} -f {} -c 'fslmerge -t {} {}'", 
        cmdPath,
        fslDir,
        outputPath,
        inputStrings
    );
    return runCommand("FSL merge", outputPath, mergeCmd);
}
