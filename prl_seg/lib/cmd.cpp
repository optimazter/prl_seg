#include "cmd.h"

std::string runCommand(const std::string &cmdName, const std::string &output, const std::string &cmd)

{
    std::cout << "Running " << cmdName << ".\n";
    if (std::filesystem::exists(output)) 
    {
        std::cout << "Output path " << output << " already exists, returning it instead. \n";
        return output;
    }
    auto c = system(cmd.c_str());
    std::cout << cmdName << " ran succesfully, output saved in " << output << ".\n";
    return output;
}