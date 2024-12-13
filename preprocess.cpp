#include <filesystem>
#include <memory>
#include <array>
#include <dataset.h>
#include <torch/torch.h>


const double isometricVoxelSize = 0.65;
const int TE = 35;
const double B0 = 3;
const double alpha = 0.1;
const double PDFTolerance = 0.1;


int main()
{     
    auto inputDir = std::filesystem::path("C:\\Users\\adria\\dev\\prl_seg\\assets\\Lou_et_al_dataset");
    auto outputDir = std::filesystem::path("C:\\Users\\adria\\dev\\prl_seg\\output");
    
    std::array<double, 3> voxelSize = {isometricVoxelSize, isometricVoxelSize, isometricVoxelSize};
    
    createLesionDataset(
        inputDir,
        outputDir,
        "phase_pp.nii.gz",
        "flair_pp.nii.gz",
        "seg2phase.nii.gz",
        "phase_unwrapped_ubmask.nii.gz"
    );


}


