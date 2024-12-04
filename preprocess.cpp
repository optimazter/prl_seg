#include <filesystem>
#include <memory>
#include <array>
#include <QSM.h>
#include <dataset.h>


const double isometricVoxelSize = 0.65;
const int TE = 35;
const double B0 = 3;
const double alpha = 0.1;
const double PDFTolerance = 0.1;


int main()
{     
    auto inputDir = std::filesystem::path("/home/adrian-hjertholm-voldseth/dev/prl_seg_master/assets/Lou_et_al_dataset");
    auto outputDir = std::filesystem::path("/home/adrian-hjertholm-voldseth/dev/prl_seg_master/output");
    
    
    std::array<double, 3> voxelSize = {isometricVoxelSize, isometricVoxelSize, isometricVoxelSize};


    std::string patientID = "001";
    auto inputT2UnwrappedPhasePath = (inputDir / patientID / "phase_pp.nii.gz").string();
    
    createLesionDataset(
        inputDir,
        outputDir,
        "phase_pp.nii.gz",
        "seg2phase.nii.gz",
        "phase_unwrapped_ubmask.nii.gz"
    );

}


