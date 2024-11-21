import os
import torch
from torch.utils.data import TensorDataset
import matplotlib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from rich.progress import Progress
from pathlib import Path


def to_nifti_filename(filename: str) -> Path:
    return Path(filename + ".nii.gz")


def nifti_to_tensor(img_path: str):
    img = sitk.ReadImage(img_path)
    img = sitk.Getnifti_arrayFromImage(img)
    return torch.Tensor(img)


class NIfTIToTensor:
    def __call__(self, img_path: str):
        return nifti_to_tensor(img_path=img_path)


class TorchNIfTIDataset(TensorDataset):

    def __init__(self, load_dir: str, save_file: str, img_name: str, label_name: str,
        img_preprocessor = NIfTIToTensor(), label_preprocessor = NIfTIToTensor()):

        if os.path.isfile(save_file):
            self = torch.load(save_file)
            return
        
        labels = []
        imgs = []
        with Progress() as progress:
            patient_ids = os.listdir(load_dir)
            dataset_task = progress.add_task('Creating nifti dataset', total = len(patient_ids))
            for patient_id in patient_ids:
                nifti_img_path = f'{load_dir}/{patient_id}/{img_name}.nii.gz'
                nifti_label_path = f'{load_dir}/{patient_id}/{label_name}.nii.gz'
                if os.path.isfile(nifti_img_path) and os.path.isfile(nifti_label_path):

                    nifti_img = sitk.ReadImage(nifti_img_path)
                    nifti_label = sitk.ReadImage(nifti_label_path)

                    nifti_img_tensor = img_preprocessor(nifti_img)
                    nifti_label_tensor = label_preprocessor(nifti_label)

                    imgs.append(nifti_img_tensor)
                    labels.append(nifti_label_tensor)
                    
                    progress.update(dataset_task, advance=1)
                    progress.refresh()
            
        
            labels_t = torch.stack(labels, dim = 0)
            imgs_t = torch.stack(imgs, dim = 0)
            super(imgs_t, labels_t)
            torch.save(self, save_file)
            print(f'Conversion completed, file saved at {save_file}')


def plot_nifti_on_ax(axis: plt.axes, path: str, title: str, idx = 100):
    ax = axis
    img = sitk.ReadImage(path)
    img_nifti_arr = sitk.GetArrayFromImage(img)
    ax.set_title(title)
    ax.imshow(img_nifti_arr[..., idx], cmap = "gray")
    plt.axis('off')


def plot_nifti(nifti_img: torch.Tensor = None, mask: torch.Tensor = None, title: str = None):
    img = nifti_img.long()
    if mask is not None:      
        img = torch.dstack((img, img, img))
        mask = torch.dstack((mask, torch.zeros_like(mask), torch.zeros_like(mask)))
        indices = mask > 0
        img[indices] = 255
        plt.imshow(img)
    else:
        plt.imshow(img, cmap = "gray")
    plt.axis('off')
    if title:
        plt.suptitle(title, fontsize = 10)
    plt.show()

