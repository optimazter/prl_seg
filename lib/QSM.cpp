#include "QSM.h"
#include <float.h>
#include <iostream>
#include <sstream>

PRLSeg::PRLSeg(const std::array<double, 3> &voxelSize, int TE, double B0, double alpha, double mu, double gyro, double pdfTol)
: voxelSize(voxelSize),
TE(TE),
B0(B0),
alpha(alpha),
mu(mu),
gyro(gyro),
phsScale(TE * gyro * B0),
pdfTol(pdfTol)
{
    matlabPtr = matlab::engine::startMATLAB();  
    try
    {
        matlabPtr->eval(u"addpath(genpath(\"qsm_pipeline\"))");
    }
    catch (const std::exception &e)
    {
        std::cout << e.what();
    }
}

void PRLSeg::SetQSMPaths(const std::string &T2PhasePath, const std::string &T2MagPath, const std::string &BETPath, const std::string &unwrappedPhasePath)
{
    this->T2PhasePath = T2PhasePath;
    this->BETPath = BETPath;
    this->unwrappedPhasePath = unwrappedPhasePath;
    this->T2MagPath = T2MagPath; 

}

void PRLSeg::ExecuteQSM(const std::filesystem::path &outputDir)
{
    assert(!(T2PhasePath.empty() && unwrappedPhasePath.empty() && BETPath.empty()) && "Need either T2 Phase or T2 Unwrapped Phase image for Brain Extraction");
    assert(!(T2PhasePath.empty() && unwrappedPhasePath.empty()) && "Need either T2 Phase or T2 Unwrapped Phase to run QSM");

    std::cout << "##########   Running QSM   ##########\n\n";

    bool juliaRunning = false;

    if (!std::filesystem::is_directory(outputDir))
    {
        std::filesystem::create_directory(outputDir);
    }

    if (BETPath.empty())
    {
        std::cout << "  -- Running FSL BET for Brain Extraction\n";
        BETPath = (outputDir / "fsl_bet.nii.gz").string();
        if (!T2PhasePath.empty())
        {
            BET(T2PhasePath, BETPath);
        }
        else if (!unwrappedPhasePath.empty())
        {
            BET(unwrappedPhasePath, BETPath);
        }
    }

    if (unwrappedPhasePath.empty())
    {
        JULIA_RUN();
        juliaRunning = true;
        std::cout << "  -- Running ROMEO for Phase Unwrapping\n";
        unwrappedPhasePath = (outputDir / "romeo_unwrapped.nii.gz").string();
        ROMEO(T2PhasePath, BETPath, unwrappedPhasePath);
    }

    NLQSM(unwrappedPhasePath, BETPath, (outputDir / "medi_lbv.nii.gz").string(),(outputDir / "fansi_qsm.nii.gz").string());

    using namespace itk::simple;
    if (juliaRunning)
    {
        try
        {
            jl_atexit_hook(0);
        }
        catch(const std::exception& e)
        {
            std::cout << e.what() << '\n';
        }
    }

}

void PRLSeg::NLQSM(const std::string &inputUnwrappedPhase,  const std::string &inputBetPath, const std::string &outputLBVPath, const std::string &outputQSMPath)
{
    using namespace matlab::engine;

    std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();

    matlab::data::ArrayFactory factory;

//function QSM(input_unwrapped_phase, input_bet_mask, output_lbv_path, output_qsm_path, TE, B0, gyro, spatial_res)

    std::vector<matlab::data::Array> args({
        factory.createCharArray(inputUnwrappedPhase),
        factory.createCharArray(inputBetPath),
        factory.createCharArray(outputLBVPath),
        factory.createCharArray(outputQSMPath),
        factory.createScalar<double>(TE),
        factory.createScalar<double>(B0),
        factory.createScalar<double>(gyro),
        factory.createArray<double>({1, 3}, voxelSize.begin(), voxelSize.end()),
    });

    try 
    {
        matlabPtr->eval(u"addpath(genpath(\"qsm_pipeline\"))");
        matlabPtr->feval(u"NLQSM", 0, args);
    } 
    catch (const std::exception &e)
    {
        std::cout << e.what();
    }

}

void PRLSeg::PDF(const std::string &inputUnwrappedPhase, const std::string &inputBetPath, const std::string &outputRDFPath)
{
    using namespace matlab::engine;

    std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
    matlab::data::ArrayFactory factory;

    //function MEDI_PDF(input_T2_phase_path, input_mask_path, output_rdf_path, voxel_size, pdf_tolerance, TE)

    std::vector<matlab::data::Array> args({
        factory.createCharArray(inputUnwrappedPhase),
        factory.createCharArray(inputBetPath),
        factory.createCharArray(outputRDFPath),
        factory.createArray<double>({1, 3}, voxelSize.begin(), voxelSize.end()),
        factory.createScalar<double>(pdfTol),
        factory.createScalar<int16_t>(TE)
    });

    try 
    {
        matlabPtr->eval(u"addpath(genpath(\"qsm_pipeline\"))");
        matlabPtr->feval(u"MEDI_PDF", 0, args);
    } 
    catch (const std::exception &e)
    {
        std::cout << e.what();
    }

}

void PRLSeg::DipoleInversion(const std::string &inputRDFPath, const std::string &outputDIPath)
{

    matlab::data::ArrayFactory factory;

    //FANSI_DI(input_rdf_path, input_mask_path, output_di_path, alpha1, mu1, phs_scale, spatial_res)
    std::vector<matlab::data::Array> args({
        factory.createCharArray(inputRDFPath),
        factory.createCharArray(outputDIPath),
        factory.createScalar<double>(alpha),
        factory.createScalar<double>(mu),
        factory.createScalar<double>(phsScale),
        factory.createArray<double>({1, 3}, voxelSize.begin(), voxelSize.end())
    });

    try 
    {
        matlabPtr->eval(u"addpath(genpath(\"qsm_pipeline\"))");
        matlabPtr->feval(u"FANSI_DI", 0, args);
    } 
    catch (const std::exception &e)
    {
        std::cout << e.what();
    }
}


void PRLSeg::ROMEO(const std::string &inputT2PhasePath, const std::string &inputMaskPath, const std::string &outputPath)
{
    runROMEO(inputT2PhasePath.c_str(), inputMaskPath.c_str(), outputPath.c_str());
}

void PRLSeg::BET(const std::string & inputPath, const std::string & outputPath)
{
    std::string betCmd;  
    std::stringstream s;
    s << "bash" << FSL_CMD_PATH << " '-f " << FSL_DIR << " -c bet " <<  inputPath << " " << outputPath << "'";
    auto c = system(betCmd.c_str());
}

void PRLSeg::ImageToMask(const std::string &inputPath, const std::string &outputPath)
{
    using namespace itk::simple;
    Image inputImage = ReadImage(inputPath, sitkFloat32);
    Image mask = BinaryThreshold(inputImage, 1e-6, DBL_MAX, 1, 0);
    Image maskBin = Cast(mask, sitkFloat32);
    WriteImage(maskBin, outputPath);

}
