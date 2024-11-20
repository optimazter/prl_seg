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

const std::filesystem::path INPUT_DIR = std::filesystem::path("/home/adrian-hjertholm-voldseth/dev/prl_seg/assets/DB1");
const std::filesystem::path OUTPUT_DIR = std::filesystem::path("/home/adrian-hjertholm-voldseth/dev/prl_seg/output");

const int NUM_ECHOS = 5;

void run(const std::filesystem::path &inDir, const std::filesystem::path  &outDir)
{
    if (!std::filesystem::is_directory(outDir))
    {
        std::filesystem::create_directory(outDir);
    }

    std::vector<std::string> echos;
    echos.reserve(NUM_ECHOS);

    FSL fsl("/usr/local/fsl");
    MEDI medi("/usr/local/MATLAB/R2024b");
    ROMEO romeo;
    

    for (int i = 1; i < NUM_ECHOS + 1; i++)
    {
        auto echoInputPhase = (inDir / std::format("sub-001_part-phase_MEGRE_echo-{}.nii.gz", i)).generic_string(); 
        auto echoInputMag = (inDir / std::format("sub-001_part-mag_MEGRE_echo-{}.nii.gz", i)).generic_string(); 

        auto echoMask = (outDir / std::format("bet_echo-{}.nii.gz", i)).generic_string(); 
        fsl.BET(echoInputPhase, echoMask);

        auto echoCorrectedPhase = (outDir / std::format("n4_corrected_phase_echo-{}.nii.gz", i)).generic_string(); 
        SimpleITKUtils::applyN4BiasCorrection(echoInputPhase, echoMask, echoCorrectedPhase, 2);

        auto echoCorrectedMag = (outDir / std::format("n4_corrected_mag_echo-{}.nii.gz", i)).generic_string(); 
        SimpleITKUtils::applyN4BiasCorrection(echoInputMag, echoMask, echoCorrectedMag, 2);

        auto echoUnwrapped = (outDir / std::format("phase_unwrapped_echo-{}.nii.gz", i)).generic_string(); 
        auto test = romeo.romeo(echoCorrectedPhase.c_str(), echoCorrectedMag.c_str(), echoUnwrapped.c_str());

        echos.push_back(echoUnwrapped);
    
    }
    auto echosCombined = fsl.merge(echos, (outDir / "echos_merged.nii.gz").generic_string());
    auto betMask = fsl.BET(echosCombined, (outDir / "bet_merged.nii.gz").generic_string());
    auto backgroundFieldRemoval = medi.PDF(echosCombined, betMask, (outDir / "pdf_merged.nii.gz").generic_string());


}

int main()
{
    auto inputDir = INPUT_DIR / "sub-001/anat";
    auto outputDir = OUTPUT_DIR / "sub-001";
    run(inputDir, outputDir);

    jl_atexit_hook(0);
    return 0;

}