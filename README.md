# 1 - About

This repository contains all the code used in my master thesis in Medical Technology at the University of Bergen: *AI Driven Paramagnetic Rim Lesion Differentiation in Multiple Sclerosis* by Adrian Hjertholm Voldseth [[1]](#1). Feel free to contact me at adrian@relu.no if you have any questions.

# 2 - Overview

The source code used for the preprocessing, training and visualization is available in the [pylib](https://github.com/optimazter/prl_seg/tree/main/pylib) folder.

An overview of most important files in the pylib source tree is given below:

* models/
    * [unet.py](https://github.com/optimazter/prl_seg/blob/main/pylib/models/unet.py): The PRLU-Net created and described in the thesis [[1]](#1).
    * [unet3d.py](https://github.com/optimazter/prl_seg/blob/main/pylib/models/unet3d.py): 3D version of the PRLU-Net.
    * [resnet.py](https://github.com/optimazter/prl_seg/blob/main/pylib/models/resnet.py): ResNet which was ultimately not used.
    * [resnet3d.py](https://github.com/optimazter/prl_seg/blob/main/pylib/models/resnet3d.py): 3D version of the ResNet.
* imaging/
    * [transforms.py](https://github.com/optimazter/prl_seg/blob/main/pylib/imaging/transforms.py): Image transformation tools.
    * [lesion_tools.py](https://github.com/optimazter/prl_seg/blob/main/pylib/imaging/lesion_tools.py) Tools used in the image processing task specific to the lesion masks. Such as growing central lines of PRLs to lesion masks. 
* datasets/
    * [star.py](https://github.com/optimazter/prl_seg/blob/main/pylib/datasets/star.py): Final dataset creation pipeline used in this thesis.
* evaluation/
    * [evaluation.py](https://github.com/optimazter/prl_seg/blob/main/pylib/evaluation/evaluation.py) as described in the thesis.
* [visualization.py](https://github.com/optimazter/prl_seg/blob/main/pylib/visualization.py): Visualization tools and Tkinter application for N number of 3D images with or without masks.
* [training.py](https://github.com/optimazter/prl_seg/blob/main/pylib/training.py): Training and validation pipeline used in the thesis for the training of PRLU-Net.


The [pylib](https://github.com/optimazter/prl_seg/tree/main/pylib) tools was ultimately implemented in the two notebooks: [training.ipynb](https://github.com/optimazter/prl_seg/tree/main/training.ipynb) and [visualization.ipynb](https://github.com/optimazter/prl_seg/tree/main/visualization.ipynb).

All paths are set in the [config.py](https://github.com/optimazter/prl_seg/tree/main/config.py) file.

# 3 - Data

The code in this repository uses the dataset obtained by Lou et al. [[2]](#2) available at [the National Institute of Health](https://data.ninds.nih.gov/Reich/Loe/index.html). For the notebooks in the repository to run, the dataset must be downloaded and placed in a folder named *assets*. 

# 4 - References
<a id="1">[1]</a>
Voldseth H. Adrian. (2025) AI Driven Paramagnetic Rim Lesion Differentiation in Multiple Sclerosis.
<a id="2">[2]</a>  
Lou, C., Sati, P., Absinta, M., Clark, K., Dworkin, J. D., Valcarcel, A. M., Schindler, M. K., Reich, D. S., Sweeney, E. M., & Shinohara, R. T. (2021). Fully automated detection of paramagnetic rims in multiple sclerosis lesions on 3T susceptibility-based MR imaging. *NeuroImage: Clinical*, 31, 102796. https://doi.org/10.1016/j.nicl.2021.102796

