import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
import matplotlib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from rich.progress import Progress
from pathlib import Path
from enum import Enum
from collections import deque
from pylib.singleton import Singleton

def to_nifti_filename(filename: str) -> Path:
    return Path(filename + ".nii.gz")


def nifti_to_tensor(img_path: str, dtype = torch.float32):
    img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img)
    return torch.Tensor(img)


def bin_to_multiclass_mask(bin_mask: torch.Tensor, bbin_mmask: torch.Tensor) -> torch.Tensor:
    #Convert binary mask for binary mask from 1 to 2
    bbin_mmask = bbin_mmask * 2
    bbin_mmask = bbin_mmask * bin_mask


def region_growing(mask, seed):
    h, w = mask.shape

    grown_region = torch.zeros_like(mask, dtype=torch.bool)

    queue = deque([seed])

    connectivity = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()
        if not (0 <= x < h and 0 <= y < w):
            continue
        if mask[x, y] and not grown_region[x, y]:
            grown_region[x, y] = True

            for dx, dy in connectivity:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not grown_region[nx, ny]:
                    queue.append((nx, ny))

    return grown_region




class NIfTIToTensor:
    def __call__(self, img_path: str):
        return nifti_to_tensor(img_path=img_path)


def create_lesion_dataset(load_dir: str, save_file: str, img_path: str, all_lesions_path: str, prl_path: str = None):
    
    all_lesions_lst = []
    prls_lst = []
    img_lst = []

    with Progress() as progress:

        patient_ids = os.listdir(load_dir)
        dataset_task = progress.add_task('Loading images', total = len(patient_ids))

        for patient_id in patient_ids:
            img_full_path = f'{load_dir}/{patient_id}/{img_path}'
            all_lesions_full_path = f'{load_dir}/{patient_id}/{all_lesions_path}'
            prl_full_path  = f'{load_dir}/{patient_id}/{prl_path}'

            assert(os.path.isfile(img_full_path) and os.path.isfile(all_lesions_full_path))
            if prl_path : assert(os.path.isfile(prl_full_path))

            img = nifti_to_tensor(img_full_path)
            all_lesions = nifti_to_tensor(all_lesions_full_path, dtype= torch.long)
            prls = nifti_to_tensor(prl_full_path, dtype=torch.long) if prl_path else None

            img_lst.append(img)
            all_lesions_lst.append(all_lesions)
            if prl_path: prls_lst.append(prls)
            
            progress.update(dataset_task, advance=1)
            progress.refresh()
        
        imgs_t = torch.cat(img_lst, dim = -1).permute(2, 0, 1)
        all_lesions_t = torch.cat(all_lesions_lst, dim = -1).permute(2, 0, 1)

        print(f"Loaded {imgs_t.shape[0]} individual images with resolution {imgs_t.shape[1]} x {imgs_t.shape[2]}")

        assert(imgs_t.shape == all_lesions_t.shape)

        if prl_path:
            prls_t = torch.cat(prls_lst, dim = -1).permute(2, 0, 1)
            assert(imgs_t.shape == prls_t.shape)

            rg_task = progress.add_task('Region Growing to combine masks', total = prls_t.shape[0])

            for i in range(prls_t.shape[0]):
        
                seed_points = torch.nonzero(prls_t[i].bool(), as_tuple=False)

                for seed in seed_points:
                    grown_region = region_growing(all_lesions_t[i], seed.tolist())
                    prls_t[i][grown_region] = 2
                
                progress.update(rg_task, advance=1)
                progress.refresh()

            dataset = TensorDataset(imgs_t, prls_t.long())
        else:
            dataset = TensorDataset(imgs_t, all_lesions_t.long())

        torch.save(dataset, save_file)
        print(f'Conversion completed, file saved at {save_file}')
        return dataset










