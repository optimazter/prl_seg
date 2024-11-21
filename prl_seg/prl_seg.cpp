
#include <julia.h>
#include <sitk_utils.h>
#include <iostream>
#include <string>
#include <format>
#include <filesystem>
#include <stdlib.h>

#include <fsl.h>
#include <medi.h>
#include <romeo.h>
#include <prl_seg.h>



void PRLSeg::QSM(const std::filesystem::path &inputPhasePath, const std::filesystem::path  &outDir)
{

    if (!std::filesystem::is_directory(outDir))
    {
        std::filesystem::create_directory(outDir);
    }

    FSL fsl("/usr/local/fsl");
    MEDI medi("/usr/local/MATLAB/R2024b");
    ROMEO romeo;    

    auto binMask = fsl.BET(inputPhasePath, (outDir / "bet.nii.gz").string());
    auto unwrapped = romeo.romeo(inputPhasePath, (outDir / "romeo.nii.gz").string());
    auto RDF = medi.PDF(unwrapped, binMask, (outDir / "pdf.nii.gz").string());

    jl_atexit_hook(0);

}
