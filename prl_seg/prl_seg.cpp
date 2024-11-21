
#include <julia.h>
#include <sitk_utils.h>
#include <iostream>
#include <string>
#include <format>
#include <vector>
#include <filesystem>
#include <stdlib.h>

#include <fsl.h>
#include <medi.h>
#include <romeo.h>
#include <prl_seg.h>



void PRLSeg::QSM(const std::filesystem::path &inputPhasePath, const std::filesystem::path &outDir, const double &voxelSize, const double &voxelSizeY, const double &voxelSizeZ)
{

    double x = voxelSize;
    double y = voxelSizeY == 0 ? voxelSize : voxelSizeY;
    double z = voxelSizeZ == 0 ? voxelSize : voxelSizeZ;

    std::array<double, 3> voxelSizeVec = {x, y, z};


    if (!std::filesystem::is_directory(outDir))
    {
        std::filesystem::create_directory(outDir);
    }

    FSL fsl("/usr/local/fsl");
    MEDI medi("/usr/local/MATLAB/R2024b");
    ROMEO romeo;    

    auto binMask = fsl.BET(inputPhasePath, (outDir / "bet.nii.gz").string());

    auto unwrapped = romeo.romeo(inputPhasePath, binMask, (outDir / "romeo.nii.gz").string());
    auto RDF = medi.PDF(unwrapped, binMask, (outDir / "pdf.nii.gz").string(), voxelSizeVec);

    jl_atexit_hook(0);

}
