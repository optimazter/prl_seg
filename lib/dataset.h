#pragma once

#include <torch/torch.h>
#include <SimpleITK.h>
#include <string>
#include <filesystem>
#include <vector>
#include <iostream>


#define DIM_D  0
#define DIM_N  0
#define DIM_H  1
#define DIM_W  2


namespace fs = std::filesystem;
namespace sitk = itk::simple;

struct Point 
{
    int x, y;
};

void createLesionDataset(
    const fs::path &loadDir, 
    const fs::path &saveDir, 
    const std::string &imgfilename, 
    const std::string &allLesionsfilename, 
    const std::string &PRLfilename, 
    const std::vector<float> trainValTestSplit = {0.8, 0.1, 0.1}
);


torch::Tensor NIfTIToTensor(const fs::path &path, sitk::PixelIDValueEnum dtype);


torch::Tensor regionGrow(const torch::Tensor &tensor, const torch::Tensor &seed);
