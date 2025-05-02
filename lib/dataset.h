#pragma once

#include <torch/torch.h>
#include <string>
#include <filesystem>
#include <vector>
#include <iostream>
#include <SimpleITK.h>


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
    const std::string &T2PhaseFileName, 
    const std::string &T2FlairFileName,
    const std::string &allLesionsFileName, 
    const std::string &PRLFileName
);

void loadTensors(
    const fs::path &loadDir, 
    torch::Tensor &phaseTen, 
    torch::Tensor &flairTen,
    torch::Tensor &allLesionsTen, 
    torch::Tensor &PRLsTen,
    const std::string &T2PhaseFileName, 
    const std::string &T2FlairFileName,
    const std::string &allLesionsFileName, 
    const std::string &PRLFileName
);

void cropTensors(
    torch::Tensor &phaseTen, 
    torch::Tensor &flairTen,
    torch::Tensor &allLesionsTen, 
    torch::Tensor &PRLsTen
);

void flipTensors(
    torch::Tensor &phaseTen, 
    torch::Tensor &flairTen,
    torch::Tensor &allLesionsTen, 
    torch::Tensor &PRLsTen
);

void growAllRegions(
    torch::Tensor &allLesionsTen, 
    torch::Tensor &PRLsTen 
);

void normalizeTensor(
    torch::Tensor &tensor
);

torch::Tensor NIfTIToTensor(const fs::path &path, sitk::PixelIDValueEnum dtype);


torch::Tensor regionGrow(const torch::Tensor &tensor, const torch::Tensor &seed);
