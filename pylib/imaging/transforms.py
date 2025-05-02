import SimpleITK as sitk
import torch
from typing import List, Tuple, Type, Union
import numpy as np
import math

def load_nifti_to_tensors(nifti_path: str):
    img = sitk.ReadImage(nifti_path)
    ten = torch.tensor(sitk.GetArrayFromImage(img)).permute(2, 1, 0)
    return ten



def crop_tensor(tensor: torch.Tensor):
    n, w, h = tensor.shape
    tensor = tensor[:, :, h - w : h]
    return tensor


def crop_and_pad(image: torch.Tensor, center: Type[Union[Tuple[int], List[int]]], new_size: Type[Union[Tuple[int], List[int]]]) -> torch.Tensor:
    """
    Crop and pad a 2D or 3D image tensor to a new size centered around the given center.

    Args:
        image: A 2D or 3D torch.Tensor of shape (H, W) or (D, H, W).
        center: A tuple or list specifying the center coordinates (x, y) for 2D or (z, x, y) for 3D.
        new_size: A tuple or list specifying the new size (new_h, new_w) for 2D or (new_d, new_h, new_w) for 3D.

    Returns:
        A cropped and padded tensor of the specified new size.
    """
    if len(image.shape) == 2:  # 2D case
        x, y = center
        new_h, new_w = new_size
        H, W = image.shape

        # Create an empty tensor with the new shape
        new_img = torch.zeros((new_h, new_w), dtype=image.dtype, device=image.device)

        # Calculate the start and end indices for the new tensor
        start_x = max(0, new_h // 2 - x)
        end_x = min(new_h, new_h // 2 + (H - x))
        start_y = max(0, new_w // 2 - y)
        end_y = min(new_w, new_w // 2 + (W - y))

        # Calculate the corresponding start and end indices for the original image
        orig_start_x = max(0, x - new_h // 2)
        orig_end_x = min(H, x + (new_h - new_h // 2))
        orig_start_y = max(0, y - new_w // 2)
        orig_end_y = min(W, y + (new_w - new_w // 2))

        # Copy the image to the new tensor
        new_img[start_x:end_x, start_y:end_y] = image[orig_start_x:orig_end_x, orig_start_y:orig_end_y]

    elif len(image.shape) == 3:  # 3D case
        z, x, y = center
        new_d, new_h, new_w = new_size
        D, H, W = image.shape

        # Create an empty tensor with the new shape
        new_img = torch.zeros((new_d, new_h, new_w), dtype=image.dtype, device=image.device)

        # Calculate the start and end indices for the new tensor
        start_z = max(0, new_d // 2 - z)
        end_z = min(new_d, new_d // 2 + (D - z))
        
        start_x = max(0, new_h // 2 - x)
        end_x = min(new_h, new_h // 2 + (H - x))

        start_y = max(0, new_w // 2 - y)
        end_y = min(new_w, new_w // 2 + (W - y))

        # Calculate the corresponding start and end indices for the original image
        orig_start_z = max(0, z - new_d // 2)
        orig_end_z = min(D, z + (new_d - new_d // 2))

        orig_start_x = max(0, x - new_h // 2)
        orig_end_x = min(H, x + (new_h - new_h // 2))

        orig_start_y = max(0, y - new_w // 2)
        orig_end_y = min(W, y + (new_w - new_w // 2))

        # Copy the image to the new tensor
        new_img[start_z:end_z, start_x:end_x, start_y:end_y] = image[orig_start_z:orig_end_z, orig_start_x:orig_end_x, orig_start_y:orig_end_y]

    else:
        raise ValueError("Input image must be either 2D or 3D.")

    return new_img




def create_phase_mask(phase, n = 4) -> torch.Tensor:
    """
    Create a phase mask for the given phase tensor. Assumes the phase is between [-pi, pi].
    """
    mask = torch.where(
        phase > 0,
        1 - phase  / phase.max(),
        torch.tensor(1, device=phase.device, dtype=phase.dtype)
    )

    return torch.pow(mask, n)



def high_pass_filter(image: torch.Tensor, cutoff: float = 0.1) -> torch.Tensor:
    assert image.dim() == 2, "Image must be 2D of shape (H, W)"
    fft_img = torch.fft.fft2(image)
    fft_shift_img = torch.fft.fftshift(fft_img)

    H, W= fft_shift_img.shape # height and width
    cy, cx = int(H/2), int(W/2) # centerness
    rh, rw = int(cutoff * cy), int(cutoff * cx) # filter_size

    # the value of center pixel is zero.
    fft_shift_img[cy-rh:cy+rh, cx-rw:cx+rw] = 0
    # restore the frequency image
    ifft_shift_img = torch.fft.ifftshift(fft_shift_img)
    # inverce fft
    ifft_img = torch.fft.ifft2(ifft_shift_img).real
    return ifft_img



def low_pass_filter(image: torch.Tensor, kernel_size = 15) -> torch.Tensor:
    assert image.dim() == 2, "Image must be 2D of shape (H, W)"
    kernel = torch.ones((kernel_size, kernel_size), dtype=image.dtype, device=image.device) / (kernel_size * kernel_size)
    return torch.nn.functional.conv2d(image.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size//2).squeeze(0).squeeze(0)


def minimum_intensity_projection(image: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum intensity projection of a 3D tensor along the specified axis.
    """
    assert image.dim() == 3, "Input tensor must be 3D of shape (D, H, W)."

    mip = torch.min(image, dim = 0, keepdim=True)
    return mip