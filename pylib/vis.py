import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import torch 
import pylib.nifti as nii


def plot_nifti_on_ax(axis: plt.axes, img, mask = None, title: str = "", cmap = "gray", idx = 100):
    ax = axis
    if isinstance(img, str):
        img = nii.nifti_to_tensor(img)
   
    
    if (len(img.shape) == 3):
        img = img[..., idx]

    if mask is not None:
        
        if isinstance(mask, str):
            mask = nii.nifti_to_tensor(img)

        img = torch.dstack((img, img, img))

        if not isinstance(mask, list):
            mask = [mask]

        assert(len(mask) <= 3)

        for i, m in enumerate(mask):
            if (len(m.shape == 3)):
                m = m[..., idx]
            indices = m > 0
            c = [0, 0, 0]
            c[i] = 255
            img[indices] = torch.tensor(c)

    ax.imshow(img, cmap = cmap)
    ax.set_title(title)
    ax.set_axis_off()


def plot_multi_class_mask_on_ax(axis: plt.axes, mask: torch.Tensor, classes: list, colors: list, title: str = ""):
    assert(len(classes) == len(colors))
    ax = axis

    img = torch.zeros([mask.shape[0], mask.shape[1], 3], dtype = torch.long)

    for cl, color in zip(classes, colors):
        assert(len(color) == 3)
        indices = mask == cl
        img[indices] = torch.tensor(color, dtype = torch.long)

    ax.imshow(img)
    ax.set_title(title)
    ax.set_axis_off()



def plot_nifti_mask(img_path: str, mask_path: str, title: str = None, idx = 100):

    img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img)
    img_arr = img_arr[..., idx]

    mask = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask)
    mask_arr = mask_arr[..., idx]
 
    img_stack = np.dstack((img_arr, img_arr, img_arr))
    indices = mask_arr > 0
    img_stack[indices] = [255, 0, 0]

    plt.imshow(img_stack)
    plt.axis('off')
    if title:
        plt.suptitle(title, fontsize = 10)
    plt.show()