#include <filesystem>
#include <memory>
#include <array>
#include <QSM.h>


int main()
{     
    auto inputDir = std::filesystem::path("/home/adrian-hjertholm-voldseth/dev/prl_seg_master/assets/Lou_et_al_dataset");
    //auto inputDir = std::filesystem::path("/home/adrian-hjertholm-voldseth/dev/prl_seg_master/assets/DB1");
    auto outputDir = std::filesystem::path("/home/adrian-hjertholm-voldseth/dev/prl_seg_master/output");
    
    const double isometricVoxelSize = 0.65;
    std::array<double, 3> voxelSize = {isometricVoxelSize, isometricVoxelSize, isometricVoxelSize};

    const int TE = 35;
    const double B0 = 3;
    const double alpha = 0.1;
    const double PDFTolerance = 0.1;

    std::string patientID = "001";
    auto inputT2UnwrappedPhasePath = (inputDir / patientID / "phase_pp.nii.gz").string();
    
    // std::string patientID = "sub-001";
    // auto inputT2PhasePath = (inputDir / patientID / "anat/sub-001_part-phase_MEGRE_echo-1.nii.gz").string();
    // auto inputT2MagPath = (inputDir / patientID / "anat/sub-001_part-mag_MEGRE_echo-1.nii.gz").string();

    PRLSeg preprocessor = PRLSeg(voxelSize, TE, B0, alpha, PDFTolerance);

    preprocessor.SetQSMPaths(PATH_EMPTY, PATH_EMPTY, PATH_EMPTY, inputT2UnwrappedPhasePath);
    preprocessor.ExecuteQSM(outputDir / patientID);

    
}


