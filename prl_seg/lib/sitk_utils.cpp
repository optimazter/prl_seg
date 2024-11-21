#include <sitk_utils.h>
#include <SimpleITK.h>
#include <iostream>

namespace sitk = itk::simple;


std::string SimpleITKUtils::applyN4BiasCorrection(const std::string &inputPath, const std::string &inputMaskPath, const std::string &outputPath, int shrinkFactor)
{
    std::cout << "Running N4 bias correction.\n"; 
    if (std::filesystem::exists(outputPath)) 
    {
        std::cout << "Output path " << outputPath << " already exists, returning it instead. \n";
        return outputPath;
    }
    sitk::Image inputImage = sitk::ReadImage(inputPath, sitk::sitkFloat32);
    sitk::Image image = inputImage;
    sitk::Image maskImage = sitk::ReadImage(inputMaskPath, sitk::sitkUInt8);

    if (shrinkFactor != 1)
    {
        std::vector<unsigned int> shrink(inputImage.GetDimension(), shrinkFactor);
        image = sitk::Shrink(inputImage, shrink);
        maskImage = sitk::Shrink(maskImage, shrink);
    }

    sitk::N4BiasFieldCorrectionImageFilter corrector;

    sitk::Image correctedImage = corrector.Execute(image, maskImage);

    sitk::Image logBiasField = corrector.GetLogBiasFieldAsImage(inputImage);

    sitk::Image correctedImageFullRes = sitk::Divide(inputImage, sitk::Exp(logBiasField));

    sitk::WriteImage(correctedImageFullRes, outputPath);

    std::cout << "Bias correction ran succesfully, output saved in " << outputPath << ".\n";
    return outputPath;
}
