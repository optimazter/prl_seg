#pragma once

#include <string>
#include <memory>
#include <filesystem>
#include <math.h>
#include <array>
#include <iostream>
#include <env.h>
#include <SimpleITK.h>

#include <MatlabDataArray.hpp>
#include <MatlabEngine.hpp>
#include <romeo.h>


#define PATH_EMPTY  ""


class PRLSeg
{

public:

    PRLSeg(const std::array<double, 3> &voxelSize, int TE, double B0, double alpha = 3e-4, double mu = 1e-2,  double gyro = 2*M_PI*42.58, double pdfTol = 1e-1);

    void SetQSMPaths(const std::string &T2PhasePath = PATH_EMPTY, const std::string &T2MagPath = PATH_EMPTY, const std::string &BETPath = PATH_EMPTY, const std::string &unwrappedPhasePath = PATH_EMPTY);

    void ExecuteQSM(const std::filesystem::path &outputDir);



private:
    std::array<double, 3> voxelSize;
    int TE;
    int B0;
    double alpha;
    double mu;
    double gyro;
    double pdfTol;
    double phsScale;

    std::string T2MagPath;
    std::string T2PhasePath;
    std::string BETPath;
    std::string unwrappedPhasePath;


    std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr;

    const std::string FSL_CMD_PATH = "qsm_pipeline/FSL/fsl_cmd.sh";

    void NLQSM(const std::string &inputUnwrappedPhase,  const std::string &inputBetPath, const std::string &outputLBVPath, const std::string &outputQSMPath);

    void PDF(const std::string &inputUnwrappedPhase, const std::string &inputBetPath, const std::string &outputRDFPath);

    void DipoleInversion(const std::string &inputRDFPath,  const std::string &outputDIPath);

    void BET(const std::string &inputPath, const std::string &outputPath);

    void ROMEO(const std::string &inputT2PhasePath, const std::string &inputMaskPath, const std::string &outputPath);

    void ImageToMask(const std::string &inputPath, const std::string &outputPath);

};
