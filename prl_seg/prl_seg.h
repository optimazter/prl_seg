#pragma once

#include <fsl.h>
#include <medi.h>
#include <romeo.h>
#include <julia.h>
#include <sitk_utils.h>

#include <iostream>
#include <string>
#include <format>
#include <filesystem>
#include <stdlib.h>


namespace PRLSeg 
{


    void QSM(const std::filesystem::path &inputPhasePath, const std::filesystem::path &outDir, const double &voxelSize, const double &voxelSizeY = 0, const double &voxelSizeZ = 0);

}