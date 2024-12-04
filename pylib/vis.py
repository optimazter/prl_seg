import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches
import SimpleITK as sitk
import numpy as np
import torch 
from torchvision.transforms import functional as F
import pylib.nifti as nii
import numpy as np
import os
import torchviz  as vz


DEFAULT_COLORS = [mcolors.to_rgba(c, alpha=0.4) for c, _ in mcolors.BASE_COLORS.items()]

def plot_color_description_box_on_ax(axis: plt.axes, box_width = 90, box_height = 50, padding = 10, colors = [], descriptions = []):
    ax = axis

    x = ax.get_xlim()[1] - box_width - padding 
    y = ax.get_ylim()[0] - box_height - padding

    rect = patches.Rectangle((x, y), box_width, box_height, linewidth=1, edgecolor='none', facecolor="white", zorder=2)
    ax.add_patch(rect)

    # Calculate the height of each color box
    h_d = (box_height - 2 * padding) / len(colors)

    for i, (color, description) in enumerate(zip(colors, descriptions)):
        cbox_w = box_width * 0.4
        cbox_h = h_d * 0.5

        # Calculate the position of each color box
        box_x = x + padding
        box_y = y + padding + i * h_d + (h_d - cbox_h) / 2  # Center the color box within its allocated space

        # Create the color box
        color_box = patches.Rectangle((box_x, box_y), cbox_w, cbox_h, linewidth=1, edgecolor='none', facecolor=color, zorder=3)
        ax.add_patch(color_box)

        # Add the description text
        ax.text(box_x + cbox_w + padding, box_y + cbox_h / 2, description, verticalalignment='center', zorder=4)




def plot_nifti_on_ax(axis: plt.axes, img = None, mask = None, mask_colors = DEFAULT_COLORS, title: str = "", cmap = "gray", idx = None, mask_legend: list = None, k_rot90: int = 0):
    ax = axis
    C, N, W, H = 0,0,0,0

    assert(img is not None or mask is not None, "Either Image or Mask must be specified.")

    def load_tensor(img):
        imgIsFile = (isinstance(img, str) and os.path.isfile(img))
        if not imgIsFile : assert (isinstance(img, torch.Tensor), "Image must be either the path of an image, or a pytorch Tensor.")
        if imgIsFile:
            img = nii.nifti_to_tensor(img)
        return img

    if img is not None:
        img = load_tensor(img)
        is2D = len(img.shape) == 2
        is3D = len(img.shape) == 3
        assert(is2D or is3D, "Image must be a 3D NiFTI image of shape (H, W, N) or 2D of shape (W, H)")
        if is3D:
            I = idx if idx else img.shape[-1] // 2
            img = img[..., I]
        if k_rot90 != 0:
            img = torch.rot90(img, k = k_rot90)

    if mask is not None:
        mask = load_tensor(mask)
        is3D = len(mask.shape) == 3
        is4D = len(mask.shape) == 4
        assert(is3D or is4D, "Mask must be a 4D NiFTI image of shape (C, H, W, N) or 3D of shape (C, H, W)")
        if (is4D):
            I = idx if idx else mask.shape[-1] // 2
            mask = mask[..., I]

        C, H, W = mask.shape 
        #Create RGBA image
        mask_img = torch.zeros([H, W, 4], dtype = torch.long)
        for i in range(C):
            indices = mask[i] == 1
            mask_img[indices] = (torch.tensor(mask_colors[i]) * 255).to(dtype = torch.long)

        if k_rot90 != 0:
            mask_img = F.rotate(mask_img, k_rot90)
    
        print(mask_img.shape)
    
    if img is None:  
        ax.imshow(mask_img)
    elif mask is None:
        ax.imshow(img, cmap = cmap)
    else:
        ax.imshow(img, cmap = cmap)
        ax.imshow(mask_img)
    
    if mask_legend and C > 0:
        assert(len(mask_legend) == C)
        plot_color_description_box_on_ax(ax, colors=mask_colors[:C], descriptions=mask_legend)

    
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




