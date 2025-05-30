import torch
import SimpleITK as sitk
from pathlib import Path


def to_nifti_filename(filename: str) -> Path:
    """
    Converts a filename to a NIfTI file path by appending ".nii.gz".
    Args:
        filename (str): The base filename without extension.
    Returns:
        Path: The NIfTI file path.
    """
    return Path(filename + ".nii.gz")


def nifti_to_tensor(img_path: str, dtype = torch.float32):
    """
    Reads a NIfTI image from the specified path and converts it to a PyTorch tensor.
    Args:
        img_path (str): Path to the NIfTI image file.
        dtype (torch.dtype): The desired data type of the output tensor.
    Returns:
        torch.Tensor: The image data as a PyTorch tensor.
    """
    img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img)
    return torch.Tensor(img)











