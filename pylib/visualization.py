import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib import patches

import SimpleITK as sitk
import numpy as np
import torch 
from torchvision.transforms import functional as F
import pylib.nifti as nii
import numpy as np
import os
import torchviz  as vz
import types
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Iterable, List, Tuple, Union




def plot_color_description_box_on_ax(
    axis: plt.axes, 
    box_width=0.30,  # Relative width (20% of axis width)
    box_height=0.18,  # Relative height (20% of axis height)
    padding=0.02,  # Relative padding (2% of axis width/height)
    colors=[], 
    descriptions=[],
    fontsize=10,  # Font size for the description text
):
    # Calculate the position for the bottom right corner in axis coordinates
    x = 1 - box_width - padding
    y = padding

    # Create the rectangle in axis coordinates
    rect = patches.Rectangle((x, y), box_width, box_height, transform=axis.transAxes, linewidth=1, edgecolor='none', facecolor="white", zorder=2)
    axis.add_patch(rect)

    # Calculate the height of each color box
    h_d = (box_height - 2 * padding) / len(colors)

    for i, (color, description) in enumerate(zip(colors, descriptions)):
        cbox_w = box_width * 0.4
        cbox_h = h_d * 0.5

        # Calculate the position of each color box in axis coordinates
        box_x = x + padding
        box_y = y + padding + i * h_d + (h_d - cbox_h) / 2  # Center the color box within its allocated space

        # Create the color box in axis coordinates
        color_box = patches.Rectangle((box_x, box_y), cbox_w, cbox_h, transform=axis.transAxes, linewidth=1, edgecolor='none', facecolor=color, zorder=3)
        axis.add_patch(color_box)

        # Add the description text in axis coordinates
        axis.text(box_x + cbox_w + padding, box_y + cbox_h / 2, description, transform=axis.transAxes, verticalalignment='center', zorder=4, fontsize=fontsize)




def plot_nifti_on_ax(
    axis: plt.axes, 
    img = None, 
    mask = None, 
    title: str = "", 
    cmap = "gray", 
    idx = None, 
    mask_legend: list = None, 
    k_rot90: int = 0,
    alpha_mask = 0.5,
    fontsize = 10,
    drop_first_label_channel = True #We assume that the first channel is the background mask
):
    mask_colors = [mcolors.to_rgba(c, alpha=alpha_mask) for c, _ in mcolors.BASE_COLORS.items()]
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
        is2D = len(mask.shape) == 2
        is3D = len(mask.shape) == 3
        is4D = len(mask.shape) == 4
        assert(is3D or is4D, "Mask must be a 4D NiFTI image of shape (C, H, W, N) or 3D of shape (C, H, W) or 2D of shape (H, W)")
        if (is4D):
            I = idx if idx else mask.shape[-1] // 2
            mask = mask[..., I]
        if (is2D):
            mask = mask.unsqueeze(0)

        C, H, W = mask.shape 
        assert(C >= 1, "mask must have at least 1 channel")
        if drop_first_label_channel and not is2D:
            mask = mask[1:]
            C -= 1
        #Create RGBA image
        mask_img = torch.zeros([H, W, 4], dtype = torch.long)
        for i in range(C):
            indices = mask[i] == 1
            mask_img[indices] = (torch.tensor(mask_colors[i]) * 255).to(dtype = torch.long)

        if k_rot90 != 0:
            mask_img = F.rotate(mask_img, k_rot90)

    if mask_legend and C > 0:
        assert(len(mask_legend) == C)
        plot_color_description_box_on_ax(ax, colors=mask_colors[:C], descriptions=mask_legend, fontsize=fontsize)

    if img is not None:
        ax.imshow(img, cmap=cmap)
    if mask is not None:
        ax.imshow(mask_img)
        

    ax.set_title(title)
    ax.set_axis_off()





def plot_nifti_mask(
    img_path: str, 
    mask_path: str, 
    title: str = None, 
    idx = 100
):

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





def plot_loss(f):

    assert(isinstance(f, types.GeneratorType), "Decorated function must be a generator which yields the training and validation loss during training")

    def wrapper(*args, **kwargs):

        fig, ax = plt.subplots()
        train_line = ax.plot([], [], label="Training Loss")
        val_line= ax.plot([], [], label="Validation Loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid()
        plt.show()

        all_train_loss = []
        all_val_loss = []
        train_iter = []
        val_iter = []

        for output in f(*args, **kwargs):

            assert(isinstance(output, tuple) and len(output) == 2, "Output of function must be a tuple of yielded loss and a description: 'Train' or 'Val' ")
            loss, desc = output
            assert(isinstance(loss, float) and desc in ["Train", "Val"])

            if desc == "Train":
                all_train_loss.append(loss)
                train_iter.append(len(all_train_loss))
            elif desc == "Val":
                all_val_loss.append(loss)
                val_iter.append(len(all_val_loss))

            train_line.set_data(iter, all_train_loss)
            val_line.set_data(iter, all_val_loss)

            ax.relim()
            ax.autoscale_view()

            plt.draw()
            plt.pause(0.001)

        return all_train_loss, all_val_loss

    return wrapper




def run_prediction_app(pred_data: Iterable[torch.Tensor], canvas_size: tuple = (800, 600), ):

    assert(isinstance(pred_data, Iterable), "pred_data must be an iterable of tuples (mag, phase, pred, label)")
    assert(all(isinstance(x, torch.Tensor) and len(x.shape) == 4 for x in pred_data), "pred_data must be an iterable of tuples (mag, phase, pred, label) where each element is a 4D tensor")
    idx = 0

    def update_plot():
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        mag, phase, pred, label = pred_data[idx]
        mag = mag.squeeze(0)
        phase = phase.squeeze(0)
        plot_nifti_on_ax(axs[0], img=mag, title="SWI image")
        plot_nifti_on_ax(axs[1], img=mag, mask=label, title="Target Lesions on SWI", mask_legend=["Lesion", "PRL"], drop_first_label_channel=True)
        plot_nifti_on_ax(axs[2], img=mag, mask=pred, title="Predicted Lesions on SWI", mask_legend=["Lesion", "PRL"], drop_first_label_channel=True)
        canvas.draw()

    def on_key(event):
        if event.keysym == "Right":
            idx = min(idx + 1, len(pred_data) - 1)
        elif event.keysym == "Left":
            idx = max(idx - 1, 0)
        update_plot()


    # Create the Tkinter window
    root = tk.Tk()
    root.title("Predictions Visualization")

    # Create the matplotlib figure and axes
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(canvas_size[0] / 100, canvas_size[1] / 100), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    # Bind the arrow keys to the window
    root.bind("<Left>", on_key)
    root.bind("<Right>", on_key)

    # Initialize the plot
    update_plot()

    # Start the Tkinter main loop
    root.mainloop()