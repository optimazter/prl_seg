#include "dataset.h"
#include <SimpleITK.h>

void createLesionDataset(
    const fs::path &loadDir, 
    const fs::path &saveDir, 
    const std::string &T2PhaseFileName, 
    const std::string &T2FlairFileName,
    const std::string &allLesionsFileName, 
    const std::string &PRLFileName)
{
    assert(fs::is_directory(loadDir) && fs::is_directory(saveDir));

    torch::Tensor phaseTen;
    torch::Tensor flairTen;
    torch::Tensor allLesionsTen;
    torch::Tensor PRLsTen;

    loadTensors(loadDir, phaseTen, flairTen, allLesionsTen, PRLsTen, T2PhaseFileName, T2FlairFileName, allLesionsFileName, PRLFileName);

    assert(phaseTen.sizes() == allLesionsTen.sizes() && phaseTen.sizes() == flairTen.sizes() && phaseTen.sizes() == PRLsTen.sizes());

    auto shape = phaseTen.sizes();
    unsigned int N = (int)shape[DIM_N];
    unsigned int W = (int)shape[DIM_W];
    unsigned int H = (int)shape[DIM_H];

    printf("Loaded %i individual images with resolution (%i x %i).\n", N, W, H);

    //We crop the images from the bottom!
    cropTensors(phaseTen, flairTen, allLesionsTen, PRLsTen);

    //Flip images in Height dimension
    flipTensors(phaseTen, flairTen, allLesionsTen, PRLsTen);


    std::cout << "Running region growing algorithm to expand PRLs to Lesion mask\n";
    growAllRegions(allLesionsTen, PRLsTen);


    //Normalize images
    normalizeTensor(phaseTen);
    normalizeTensor(flairTen);

    //Add a channel dimension to the images
    phaseTen = phaseTen.unsqueeze(0);
    flairTen = flairTen.unsqueeze(0);

    torch::save({flairTen, allLesionsTen}, (saveDir / "lesion_segmentation_dataset.pt").string());
    torch::save({phaseTen, PRLsTen}, (saveDir / "prl_classification_dataset.pt").string());
    std::cout << "Dataset creation was successful!\n";

}

void loadTensors(const fs::path &loadDir, 
    torch::Tensor &phaseTen, 
    torch::Tensor &flairTen, 
    torch::Tensor &allLesionsTen, 
    torch::Tensor &PRLsTen, 
    const std::string &T2PhaseFileName, 
    const std::string &T2FlairFileName,
    const std::string &allLesionsFileName, 
    const std::string &PRLFileName)
{
std::vector<torch::Tensor> allLesions;
    std::vector<torch::Tensor> phaseImages;
    std::vector<torch::Tensor> flairImages;
    std::vector<torch::Tensor> PRLs;

    std::cout << "Reading images in given directory " << loadDir << "\n";

    int i = 1;
    for (const auto& patientEntry : fs::directory_iterator(loadDir)) 
    {
        fs::path phasePath = loadDir / patientEntry / T2PhaseFileName;
        fs::path flairPath = loadDir / patientEntry / T2FlairFileName;
        fs::path allLesionsPath = loadDir / patientEntry / allLesionsFileName;
        fs::path PRLPath = loadDir / patientEntry / PRLFileName;

        if (!(fs::is_regular_file(phasePath) && fs::is_regular_file(flairPath) && fs::is_regular_file(allLesionsPath) && fs::is_regular_file(PRLPath)))
        {
            std::cout << patientEntry << " is not a valid entry!\n";
            continue;
        }

        phaseImages.push_back(NIfTIToTensor(phasePath, sitk::sitkFloat32));
        flairImages.push_back(NIfTIToTensor(flairPath, sitk::sitkFloat32));
        allLesions.push_back(NIfTIToTensor(allLesionsPath, sitk::sitkUInt8));
        PRLs.push_back(NIfTIToTensor(PRLPath, sitk::sitkUInt8));

        std::cout << "\rLoading image:  " << i++ << " \n";
        std::cout.flush(); 
    }

    std::cout << std::endl;

    phaseTen = torch::cat(phaseImages);
    flairTen = torch::cat(flairImages);
    allLesionsTen = torch::cat(allLesions);
    PRLsTen = torch::cat(PRLs);
}

void cropTensors(torch::Tensor &phaseTen, torch::Tensor &flairTen, torch::Tensor &allLesionsTen, torch::Tensor &PRLsTen)
{
    auto shape = phaseTen.sizes();
    unsigned int N = (int)shape[DIM_N];
    unsigned int W = (int)shape[DIM_W];
    unsigned int H = (int)shape[DIM_H];

    //We crop the images from the bottom!
    phaseTen = phaseTen.slice(DIM_H, H - W, H);
    flairTen = flairTen.slice(DIM_H, H - W, H);
    allLesionsTen = allLesionsTen.slice(DIM_H, H - W, H);
    PRLsTen = PRLsTen.slice(DIM_H, H - W, H);
}

void flipTensors(torch::Tensor &phaseTen, torch::Tensor &flairTen, torch::Tensor &allLesionsTen, torch::Tensor &PRLsTen)
{
    //Flip images in Height dimension
    phaseTen = torch::flip(phaseTen, {DIM_H});
    flairTen = torch::flip(flairTen, {DIM_H});
    allLesionsTen = torch::flip(allLesionsTen, {DIM_H});
    PRLsTen = torch::flip(PRLsTen, {DIM_H});
}

void growAllRegions(torch::Tensor &allLesionsTen, torch::Tensor &PRLsTen)
{
    for (int i = 0; i < PRLsTen.sizes()[DIM_N]; i++)
    {
        torch::Tensor seedPts = torch::nonzero(PRLsTen[i].to(torch::kBool));
        for (int j = 0; j < seedPts.sizes()[0]; j++)
        {
            torch::Tensor grownRegion = regionGrow(allLesionsTen[i], seedPts[j]);
            PRLsTen.index_put_({i, grownRegion}, 1);
        }
    }
}

void normalizeTensor(torch::Tensor &tensor)
{
    float mean = torch::mean(tensor).item<float>();
    float std = torch::std(tensor).item<float>();

    torch::data::transforms::Normalize<> normalizeTransform(mean, std);
    tensor = normalizeTransform(tensor);
}

torch::Tensor NIfTIToTensor(const fs::path &path, sitk::PixelIDValueEnum dtype)
{
    torch::Tensor tensor;
    sitk::Image img = sitk::ReadImage(path.string(), dtype);

    // Depth for these images are read as width by SimpleITK
    auto shape = img.GetSize();

    //These are the dimensions read by SimpleITK
    //We will convert from (D, W, H) to expected dimension by PyTorch: (D, H, W)
    unsigned int D = shape[0];
    unsigned int W = shape[1];
    unsigned int H = shape[2];


    if (img.GetPixelIDValue() == sitk::sitkFloat32)
    {
        //(D, H, W) from (D, W, H)
        tensor = torch::zeros({D, H, W}, torch::kFloat32);

        auto a = tensor.accessor<float, 3>();

        for (unsigned int d = 0; d < D; d++)
        {
            for (unsigned int h = 0; h < H; h++)
            {
                for (unsigned int w = 0; w < W; w++)
                {
                    a[d][h][w] = img.GetPixelAsFloat({d, w, h}); 
                }
            }
        }
    }

    else if (img.GetPixelIDValue() == sitk::sitkUInt8)
    {
        //(D, H, W) from (D, W, H)
        tensor = torch::zeros({D, H, W}, torch::kUInt8);

        auto a = tensor.accessor<uint8_t, 3>();

        for (unsigned int d = 0; d < D; d++)
        {
            for (unsigned int h = 0; h < H; h++)
            {
                for (unsigned int w = 0; w < W; w++)
                {
                    a[d][h][w] = img.GetPixelAsUInt8({d, w, h}); 
                }
            }
        }
    }
    else
    {
        throw std::runtime_error("Invalid dtype");
    }


    return tensor;
    
}

torch::Tensor regionGrow(const torch::Tensor &tensor, const torch::Tensor &seed)
{
    int H = tensor.size(0);
    int W = tensor.size(1);

    // Create an empty mask for the grown region
    torch::Tensor grownRegion = torch::zeros_like(tensor, torch::kBool);

    // Create a queue for the region growing algorithm
    std::queue<Point> queue;

    queue.push({seed[0].item<int>(), seed[1].item<int>()});

    // Define the 4-connectivity (up, down, left, right)
    std::vector<Point> connectivity = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    while (!queue.empty()) 
    {
        Point p = queue.front();
        queue.pop();

        int x = p.x;
        int y = p.y;

        if (!(0 <= x && x < H && 0 <= y && y < W)) 
        {
            continue;
        }

        if (tensor.index({x, y}).item<bool>() && !grownRegion.index({x, y}).item<bool>()) 
        {
            grownRegion.index_put_({x, y}, true);

            for (const auto& d : connectivity) 
            {
                int nx = x + d.x;
                int ny = y + d.y;
                if (0 <= nx && nx < H && 0 <= ny && ny < W && tensor.index({nx, ny}).item<bool>() && !grownRegion.index({nx, ny}).item<bool>()) 
                {
                    queue.push({nx, ny});
                }
            }
        }
    }

    return grownRegion;
}


