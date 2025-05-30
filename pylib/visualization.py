import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib import patches
from pylib.imaging import lesion_tools as lt

import SimpleITK as sitk
import numpy as np
import torch 
from torchvision.transforms import functional as F
import pylib.nifti as nii
import numpy as np
import os
import types
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Iterable, List, Tuple, Union, Dict
from enum import Enum
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from threading import Timer


class ImageConfig3D(Enum):
    CHWD = 0
    CWHD = 1
    DCWH = 2
    DCHW = 3
    CDHW = 4
    CDWH = 5



class Image3D:

    """
    A class to represent a 3D image with optional mask.
    The image can be in different configurations, specified by the `image_config` parameter.
    The configurations are:
    - CHWD: Channels, Height, Width, Depth
    - CWHD: Channels, Width, Height, Depth
    - DCHW: Depth, Channels, Height, Width
    - DCWH: Depth, Channels, Width, Height
    - CDHW: Channels, Depth, Height, Width
    - CDWH: Channels, Depth, Width, Height
    """

    def __init__(self, 
                 data: torch.Tensor, 
                 title: str = None, 
                 image_config: ImageConfig3D = ImageConfig3D.CHWD,
                 mask: torch.Tensor = None
            ):
        assert(isinstance(data, torch.Tensor), "data must be a torch.Tensor")
        assert(len(data.shape) == 3, "data must be a 3D tensor of shape ImageConfig")
        if mask is not None:
            assert(isinstance(mask, torch.Tensor), "mask must be a torch.Tensor")
            assert(len(mask.shape) == 3, "mask must be a 4D tensor of shape [D, C, H, W] or [C, H, W, D]")
            assert(data.shape == mask.shape, "data and mask must have the same shape")

        self.data = data
        self.title = title
        self.image_config = image_config
        self.mask = mask
        if title is None:
            if mask is not None:
                self.title = "Image and Mask"
            else:
                self.title = "Image"
        self.W = self._get_width()
        self.H = self._get_height()
        self.D = self._get_depth()
    
    def _get_width(self):
        match self.image_config:
            case ImageConfig3D.CHWD:
                return self.data.shape[2]
            case ImageConfig3D.CWHD:
                return self.data.shape[1]
            case ImageConfig3D.DCHW:
                return self.data.shape[3]
            case ImageConfig3D.DCWH:
                return self.data.shape[2]
            case ImageConfig3D.CDHW:
                return self.data.shape[3]
            case ImageConfig3D.CDWH:
                return self.data.shape[2]
            case _:
                raise ValueError("Invalid image configuration")
    
    def _get_height(self):
        match self.image_config:
            case ImageConfig3D.CHWD:
                return self.data.shape[1]
            case ImageConfig3D.CWHD:
                return self.data.shape[2]
            case ImageConfig3D.DCHW:
                return self.data.shape[2]
            case ImageConfig3D.DCWH:
                return self.data.shape[3]
            case ImageConfig3D.CDHW:
                return self.data.shape[2]
            case ImageConfig3D.CDWH:
                return self.data.shape[3]
            case _:
                raise ValueError("Invalid image configuration")
    
    def _get_depth(self):
        match self.image_config:
            case ImageConfig3D.CHWD:
                return self.data.shape[3]
            case ImageConfig3D.CWHD:
                return self.data.shape[3]
            case ImageConfig3D.DCHW:
                return self.data.shape[0]
            case ImageConfig3D.DCWH:
                return self.data.shape[0]
            case ImageConfig3D.CDHW:
                return self.data.shape[1]
            case ImageConfig3D.CDWH:
                return self.data.shape[1]
            case _:
                raise ValueError("Invalid image configuration")
    
    def get_slice(self, idx: int, return_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the slice of the image at the given index.

        Parameters:
            idx (int): The index of the slice to get.
            return_mask (bool): Whether to return the mask as well. Default is False.
        """
        match self.image_config:
            case ImageConfig3D.CHWD:
                if return_mask and self.mask is not None:
                    return self.data[..., idx], self.mask[..., idx]
                elif return_mask and self.mask is None:
                    return self.data[..., idx], None
                return self.data[..., idx]
            case ImageConfig3D.CWHD:
                if return_mask and self.mask is not None:
                    return self.data[..., idx], self.mask[..., idx]
                elif return_mask and self.mask is None:
                    return self.data[..., idx], None
                return self.data[..., idx]
            case ImageConfig3D.DCHW:
                if return_mask and self.mask is not None:
                    return self.data[idx], self.mask[idx]
                elif return_mask and self.mask is None:
                    return self.data[idx], None
                return self.data[idx]
            case ImageConfig3D.DCWH:
                if return_mask and self.mask is not None:
                    return self.data[idx], self.mask[idx]
                elif return_mask and self.mask is None:
                    return self.data[idx], None
                return self.data[idx]
            case ImageConfig3D.CDHW:
                if return_mask and self.mask is not None:
                    return self.data[:, idx], self.mask[:, idx]
                elif return_mask and self.mask is None:
                    return self.data[:, idx], None
                return self.data[:, idx]
            case ImageConfig3D.CDWH:
                if return_mask and self.mask is not None:
                    return self.data[:, idx], self.mask[:, idx]
                elif return_mask and self.mask is None:
                    return self.data[:, idx], None
                return self.data[:, idx]
            case _:
                raise ValueError("Invalid image configuration")
    






def plot_color_description_box_on_ax(
    axis: plt.axes, 
    box_width=0.30,  # Relative width (20% of axis width)
    box_height=0.18,  # Relative height (20% of axis height)
    padding=0.02,  # Relative padding (2% of axis width/height)
    colors=[], 
    descriptions=[],
    fontsize=10,  # Font size for the description text
):
    """
    Plots a color description box on the given axis.
    Parameters:
        axis (plt.axes): The axis on which to plot the color description box.
        box_width (float): The width of the box in relative coordinates (0 to 1).
        box_height (float): The height of the box in relative coordinates (0 to 1).
        padding (float): The padding around the box in relative coordinates (0 to 1).
        colors (list): A list of colors for the boxes.
        descriptions (list): A list of descriptions corresponding to each color.
        fontsize (int): Font size for the description text.
    """
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
    """
    Plots a 2D or 3D NiFTI image and its mask on the given axis.
    Parameters:
        axis (plt.axes): The axis on which to plot the image and mask.
        img (torch.Tensor or str): The image to plot. Can be a 2D or 3D tensor or a path to a NiFTI file.
        mask (torch.Tensor or str): The mask to plot. Can be a 2D, 3D or 4D tensor or a path to a NiFTI file.
        title (str): The title of the plot.
        cmap (str): The colormap to use for the image.
        idx (int): The index of the slice to plot in case of 3D images. If None, the middle slice is used.
        mask_legend (list): A list of descriptions for each mask color.
        k_rot90 (int): Number of times to rotate the image by 90 degrees counter-clockwise.
        alpha_mask (float): Alpha value for the mask overlay.
        fontsize (int): Font size for the mask legend descriptions.
        drop_first_label_channel (bool): Whether to drop the first channel of the mask, assuming it is the background.
    """
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





def run_app(
    image_3d: Union[Image3D, List[Image3D]],
    canvas_size: tuple = (800, 600),
    refresh_rate_ms: int = 100,
    **kwargs
    ):
    """
    Run a Tkinter app to visualize the data.

    Parameters:
    image_3d (Image3D): The 3D image to visualize or a list of 3D images.
    canvas_size (tuple): The size of the canvas in pixels. Default is (800, 600).
    **kwargs: Additional arguments to pass to the plot_nifti_on_ax function. Read the docstring of that function for more details.
    """

    global idx
    idx = 0

    global debounce_timer
    debounce_timer = None


    if isinstance(image_3d, Image3D):
        image_3d = [image_3d]

    

    assert(isinstance(image_3d, list) and all(isinstance(x, Image3D) for x in image_3d), "image_3d must be a list of Image3D objects")

    assert(all(image.W == image_3d[0].W for image in image_3d), "All images must have the same width")
    assert(all(image.H == image_3d[0].H for image in image_3d), "All images must have the same height")
    assert(all(image.D == image_3d[0].D for image in image_3d), "All images must have the same depth")

    W, H, D = image_3d[0].W, image_3d[0].H, image_3d[0].D

    global fig
    fig, axs = plt.subplots(ncols=len(image_3d), nrows=1, figsize=(canvas_size[0] / 100, canvas_size[1] / 100), dpi=100)
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    



    def debounce_update_plot():
        global debounce_timer
        if debounce_timer is not None:
            if debounce_timer.is_alive():
                debounce_timer.cancel()
        debounce_timer = Timer(refresh_rate_ms / 1000, update_plot)
        debounce_timer.start()

    def update_slice_label():
        txt = f"Slice: {idx + 1}/{D}"
        slice_label.config(text=txt)




    def update_plot():
        global idx
        for ax in axs:
            ax.clear()
        
        for i, image in enumerate(image_3d):
            img_2d, mask_2d = image.get_slice(idx, return_mask=True)
            plot_nifti_on_ax(axs[i], img=img_2d.squeeze(0), mask=mask_2d, title=image.title, **kwargs)
        

        canvas.draw()
        update_slice_label()

    def on_key(event):
        global idx
        if event.keysym == "Right":
            idx = min(idx + 1, D - 1)
        elif event.keysym == "Left":
            idx = max(idx - 1, 0)
        debounce_update_plot()

    def on_mouse_scroll(event):
        global idx
        if event.delta > 0:  # Scroll up
            idx = max(idx - 1, 0)
        elif event.delta < 0:  # Scroll down
            idx = min(idx + 1, D - 1)
        debounce_update_plot()

    def on_input_change(*args):
        global idx
        try:
            new_idx = int(slice_input_var.get()) - 1
            if 0 <= new_idx < D:
                idx = new_idx
                update_plot()
            else:
                slice_input_var.set(str(idx + 1))  # Reset to current valid slice
        except ValueError:
            slice_input_var.set(str(idx + 1))  # Reset to current valid slice


    try:
        # Create the Tkinter window
        root = tk.Tk()
        root.title("Predictions Visualization")

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        toolbar_frame = tk.Frame(root)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        # Add a label to display the current slice
        slice_label = tk.Label(root, text="", font=("Arial", 12))
        slice_label.pack()

        # Add an input box for slice selection
        slice_input_var = tk.StringVar(value=str(idx + 1))
        slice_input_var.trace_add("write", on_input_change)
        slice_input = tk.Entry(root, textvariable=slice_input_var, font=("Arial", 12), width=5, justify="center")
        slice_input.pack()

        # Bind the arrow keys to the window
        root.bind("<Left>", on_key)
        root.bind("<Right>", on_key)

        # Bind the mouse scroll event to the window
        root.bind("<MouseWheel>", on_mouse_scroll)

        # Initialize the plot
        update_plot()

        # Start the Tkinter main loop
        root.mainloop()
    except Exception as e:
        print(f"An error occurred: {e}")
        if 'root' in locals() and root.winfo_exists():
            root.destroy()




