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

std::string SimpleITKUtils::addEchoDim(const std::string &inputPath, const std::string &outputPath)
{
    std::cout << "Adding Echo dimension.\n"; 
    if (std::filesystem::exists(outputPath)) 
    {
        std::cout << "Output path " << outputPath << " already exists, returning it instead. \n";
        return outputPath;
    }
    sitk::Image inputImage = sitk::ReadImage(inputPath, sitk::sitkFloat32);
    std::vector<unsigned int> size3D = inputImage.GetSize();

    if (size3D.size() != 3) 
    {
        std::cerr << "The input image is not 3D! Returning original image instead!" << std::endl;
        return inputPath;
    }

    std::vector<unsigned int> size4D = {size3D[0], size3D[1], size3D[2], 1};
    sitk::Image image4D(size4D, inputImage.GetPixelID());

    for (unsigned int z = 0; z < size3D[2]; ++z) 
    {
        for (unsigned int y = 0; y < size3D[1]; ++y) 
        {
            for (unsigned int x = 0; x < size3D[0]; ++x) 
            {
                image4D.SetPixelAsFloat({x, y, z, 0}, inputImage.GetPixelAsFloat({x, y, z}));
            }
        }
    }
    sitk::WriteImage(image4D, outputPath);
    std::cout << "Adding Echo dimension ran succesfully, output saved in " << outputPath << ".\n";
    return outputPath;
}
