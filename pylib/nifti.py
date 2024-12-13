import torch
import SimpleITK as sitk
from pathlib import Path


def to_nifti_filename(filename: str) -> Path:
    return Path(filename + ".nii.gz")


def nifti_to_tensor(img_path: str, dtype = torch.float32):
    img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img)
    return torch.Tensor(img)











