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


    torch::Tensor phaseTen = torch::cat(phaseImages);
    torch::Tensor flairTen = torch::cat(flairImages);
    torch::Tensor allLesionsTen = torch::cat(allLesions);
    torch::Tensor PRLsTen = torch::cat(PRLs);

    torch::Tensor backgroundTen;
    torch::Tensor labelsTen;

    assert(phaseTen.sizes() == allLesionsTen.sizes() && phaseTen.sizes() == flairTen.sizes() && phaseTen.sizes() == PRLsTen.sizes());

    auto shape = phaseTen.sizes();

    unsigned int N = (int)shape[DIM_N];
    unsigned int W = (int)shape[DIM_W];
    unsigned int H = (int)shape[DIM_H];


    printf("Loaded %i individual images with resolution (%i x %i).\n", N, W, H);


    std::cout << "Cropping images.\n";
    //We crop the images from the bottom!
    auto phaseTenCropped = phaseTen.slice(DIM_H, H - W, H);
    auto flairTenCropped = flairTen.slice(DIM_H, H - W, H);
    auto allLesionsTenCropped = allLesionsTen.slice(DIM_H, H - W, H);
    auto PRLsTenCropped = PRLsTen.slice(DIM_H, H - W, H);


    std::cout << "Flipping images.\n";
    //Flip images in Height dimension
    auto phaseTenFlipped = torch::flip(phaseTenCropped, {DIM_H});
    auto flairTenFlipped = torch::flip(flairTenCropped, {DIM_H});
    auto allLesionsTenFlipped = torch::flip(allLesionsTenCropped, {DIM_H});
    auto PRLsTenFlipped = torch::flip(PRLsTenCropped, {DIM_H});


    std::cout << "Running region growing algorithm to expand PRLs to Lesion mask\n";
    for (int i = 0; i < N; i++)
    {
        torch::Tensor seedPts = torch::nonzero(PRLsTenFlipped[i].to(torch::kBool));
        for (int j = 0; j < seedPts.sizes()[0]; j++)
        {
            torch::Tensor grownRegion = regionGrow(allLesionsTenFlipped[i], seedPts[j]);
            PRLsTenFlipped.index_put_({i, grownRegion}, 1);
        }
    }

    //Creating background from allLesions and PRLs
    backgroundTen = 1 - allLesionsTenFlipped - PRLsTenFlipped;
    labelsTen = torch::stack({backgroundTen, allLesionsTenFlipped, PRLsTenFlipped});

    //Combine Phase and FLAIR to two different channels
    auto imagesTen = torch::stack({phaseTenFlipped, flairTenFlipped});

    //Reshape images and labels to be of expected shape (N, C, H, W)
    auto imagesTenPermuted = torch::permute(imagesTen, {1, 0, 2, 3});
    auto labelsTenPermuted = torch::permute(labelsTen, {1, 0, 2, 3});

    torch::save({imagesTenPermuted, labelsTenPermuted}, (saveDir / "dataset.pt").string());
    std::cout << "Dataset creation was successful!\n";

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


