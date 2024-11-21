#include <prl_seg.h>
#include <filesystem>


int main()
{     
    auto inputDir = std::filesystem::path("/home/adrian-hjertholm-voldseth/dev/prl_seg_master/assets/Lou_et_al_dataset");
    auto outputDir = std::filesystem::path("/home/adrian-hjertholm-voldseth/dev/prl_seg_master/output");
    

    std::string patientID = "001";

    PRLSeg::QSM(inputDir / patientID / "phase_pp.nii.gz", outputDir / patientID);

}


