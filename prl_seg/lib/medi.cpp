#include "medi.h"

MEDI::MEDI(const std::string &matlabDir)
{
    this->matlabDir = matlabDir;
    cmdPath = (std::filesystem::current_path() / "libs/MEDI/matlab_cmd.sh").string();
    pdfPath = (std::filesystem::current_path() / "libs/MEDI").string();
    
}

std::string MEDI::PDF(const std::string &inputT2PhasePath, const std::string &inputMaskPath,  const std::string &outputPath, const std::array<double, 3> &voxelSize)
{
    std::string pdfCmd = std::format("MEDI_PDF('{}','{}','{}','{}','{}','{}')", 
        inputT2PhasePath, 
        inputMaskPath, 
        outputPath,
        voxelSize.at(0),
        voxelSize.at(1),
        voxelSize.at(2)
    );
    std::string cmd = std::format("{}/bin/matlab -nodisplay -nojvm -nosplash -r \"cd('{}');{};exit\"", matlabDir, pdfPath, pdfCmd);
    return runCommand("PDF", outputPath, cmd);

}