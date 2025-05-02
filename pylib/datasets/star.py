import torch
from torch.utils.data import TensorDataset, Dataset
from pylib.datasets import dataset
from typing import List, Type, Tuple, Union
from pylib.imaging import transforms
import pylib.imaging.lesion_tools as lt
from rich.progress import Progress
import numpy as np
import math
import os
from monai.transforms import (
    Rotate,
    Flip,
    Affine,
    RandGaussianNoise,
    RandRotate,
    RandFlip,
    RandAffine,
    SpatialCrop,
    NormalizeIntensity,
)
import monai.utils.type_conversion as type_conversion





def create_star_dataset(
        patient_dirs: Type[List[str]],
        flair_fname: str,
        phase_fname: str,
        lesion_fname: str,
        prl_fname: str,
        save_train_val_dir: str,
        save_test_dir: str,
        num_test_samples: int = 1,
        crop: SpatialCrop = None,
        high_pass_kernel_size: int = None,
        augment_prl: bool = True,
    ):


    with Progress() as progress:

        n_patients = len(patient_dirs)

        #Calculate mean and std of the training set
        stat_task = progress.add_task("Calculating mean and std for training images...", total=n_patients)
        mean_flair, std_flair = 0.0, 0.0
        for i, patient_dir in enumerate(patient_dirs):
            # Skip test samples
            if i < num_test_samples:
                continue
            flair_ten, phase_ten = dataset.process_patient(patient_dir, [flair_fname, phase_fname])
            if i == 0:
                mean_flair = flair_ten.mean()
                std_flair = flair_ten.std()
            else:
                mean_flair = (mean_flair * i + flair_ten.mean()) / (i + 1)
                std_flair = (std_flair * i + flair_ten.std()) / (i + 1)

            progress.update(stat_task, advance=1)
            progress.refresh()


        flair_normalizer = NormalizeIntensity(subtrahend=mean_flair, divisor=std_flair)

        progress.remove_task(stat_task)


        task = progress.add_task("Loading data", total=n_patients)
        for i, patient_dir in enumerate(patient_dirs):

            progress.update(task, description=f"Loading data for patient {patient_dir}...")
            progress.refresh()

            flair_ten, phase_ten, lesions_ten, prl_ten = dataset.process_patient(patient_dir, [flair_fname, phase_fname, lesion_fname, prl_fname])

            if phase_ten is not None and lesions_ten is not None and prl_ten is not None:

                #Normalize the images
                flair_ten = flair_normalizer(flair_ten)

                lesions_ten[lesions_ten > 0] = 1
                prl_ten[prl_ten > 0] = 1

                progress.update(task, description="Growing PRLs to lesions...")
                lesions_ten, prl_ten = lt.grow_all_regions(lesions_ten.squeeze(0), prl_ten.squeeze(0))
                lesions_ten = lesions_ten.unsqueeze(0)
                prl_ten = prl_ten.unsqueeze(0)

                #Subtract PRLs from lesions
                lesions_ten[prl_ten > 0] = 0

                background_ten = torch.ones_like(lesions_ten) - lesions_ten - prl_ten
                mask_ten = torch.stack([background_ten, lesions_ten, prl_ten], dim=1).squeeze(0)

                progress.update(task, description="Transforming data..")

                if high_pass_kernel_size is not None:
                    # Normalize the phase image between -pi and pi
                    filter_task = progress.add_task(f"Applying high pass filter to T2*-phase for slice 0 / {phase_ten.shape[1]}...", total=phase_ten.shape[1])
                    for j in range(phase_ten.shape[1]):
                        progress.update(filter_task, advance=1, description=f"Applying high pass filter to T2*-phase for slice {j + 1} / {phase_ten.shape[1]}...")
                        progress.refresh()
                        phase_ten[0, j] -= transforms.low_pass_filter(phase_ten[0, j], kernel_size=high_pass_kernel_size)

                
                    progress.remove_task(filter_task)

                phase_mask = transforms.create_phase_mask(phase_ten, n = 4)
                star_img = flair_ten * phase_mask

                star_img = crop(star_img)
                phase_ten = crop(phase_ten)
                mask_ten = crop(mask_ten)

                assert(star_img.dim() == 4), f"SWI image has {star_img.dim()} dimensions, expected 4!"
                assert(mask_ten.dim() == 4), f"Mask image has {mask_ten.dim()} dimensions, expected 4!"
                assert(phase_ten.dim() == 4), f"Phase image has {phase_ten.dim()} dimensions, expected 4!"

                #Permute from [C, D, H, W] to [D, C, H, W]
                star_img = star_img.permute(1, 0, 2, 3)
                mask_ten = mask_ten.permute(1, 0, 2, 3)
                phase_ten = phase_ten.permute(1, 0, 2, 3)


                if i < num_test_samples:
                    torch.save([star_img, phase_ten, mask_ten], f"{save_test_dir}/{i}.pt")
                    break
                else:
                    if augment_prl:
                        progress.update(task, description="Augmenting training data...")
                        augmented_prl_img, augmented_phase, augmented_prl_label = augment_prl_images([star_img, phase_ten], mask_ten, progress)
                        augmented_prl_img, augmented_phase, augmented_prl_label = torch.stack(augmented_prl_img), torch.stack(augmented_phase), torch.stack(augmented_prl_label)
                        torch.save([augmented_prl_img, augmented_phase, augmented_prl_label], f"{save_train_val_dir}/{i}.pt")
                    else:
                        torch.save([star_img, phase_ten, mask_ten], f"{save_train_val_dir}/{i}.pt")
                
                progress.update(task, advance=1)
                progress.refresh()

        progress.update(task, description="Loading data complete!")




def create_star_dataset_3d(
        patient_dirs: Type[List[str]],
        flair_fname: str,
        phase_fname: str,
        lesion_fname: str,
        prl_fname: str,
        save_train_dir: str,
        save_val_dir: str,
        save_test_dir: str,
        augment: bool = True,
        num_val_samples: int = 2,
        num_test_samples: int = 1,
        crop: SpatialCrop = None,
    ):

    if not os.path.exists(save_train_dir):
        os.makedirs(save_train_dir)
    if not os.path.exists(save_val_dir):
        os.makedirs(save_val_dir)
    if not os.path.exists(save_test_dir):
        os.makedirs(save_test_dir)


    with Progress() as progress:

        n_patients = len(patient_dirs)

        #Calculate mean and std of the training set
        stat_task = progress.add_task("Calculating mean and std for training images...", total=n_patients)
        mean, std = 0.0, 0.0
        for i, patient_dir in enumerate(patient_dirs):

            flair_ten = dataset.process_patient(patient_dir, [flair_fname])[0]
            if i == 0:
                mean = flair_ten.mean()
                std = flair_ten.std()
            else:
                mean = (mean * i + flair_ten.mean()) / (i + 1)
                std = (std * i + flair_ten.std()) / (i + 1)

            progress.update(stat_task, advance=1)
            progress.refresh()


        normalizer = NormalizeIntensity(subtrahend=mean, divisor=std)

        task = progress.add_task("Loading data", total=n_patients)
        for i, patient_dir in enumerate(patient_dirs):

            progress.update(task, description=f"Loading data for patient {patient_dir}...")
            progress.refresh()

            flair_ten, phase_ten, lesions_ten, prl_ten = dataset.process_patient(patient_dir, [flair_fname, phase_fname, lesion_fname, prl_fname])

            if phase_ten is not None and lesions_ten is not None and prl_ten is not None:
                flair_ten = flair_ten.unsqueeze(0)
                phase_ten = phase_ten.unsqueeze(0)

                #Normalize the images
                flair_ten = normalizer(flair_ten)

                lesions_ten[lesions_ten > 0] = 1
                prl_ten[prl_ten > 0] = 1

                lesions_ten, prl_ten = lt.grow_all_regions(lesions_ten.squeeze(0), prl_ten.squeeze(0))
                lesions_ten = lesions_ten.unsqueeze(0)
                prl_ten = prl_ten.unsqueeze(0)

                #Subtract PRLs from lesions
                lesions_ten[prl_ten > 0] = 0

                background_ten = torch.ones_like(lesions_ten) - lesions_ten - prl_ten
                mask_ten = torch.stack([background_ten, lesions_ten, prl_ten], dim=1).squeeze(0)

                # Normalize the phase image between -pi and pi
                phase_ten = (phase_ten / phase_ten.max()) * math.pi

                phase_mask = transforms.create_phase_mask(phase_ten, n = 4)
                star_img = flair_ten * phase_mask
                star_img = star_img.squeeze(0)

                star_img = crop(star_img)
                mask_ten = crop(mask_ten)

                assert(len(star_img.shape) == 4)
                assert(len(mask_ten.shape) == 4)

                if i < num_val_samples:
                    torch.save([star_img, mask_ten], f"{save_val_dir}/{i}.pt")
                elif i < num_val_samples + num_test_samples:
                    torch.save([star_img, mask_ten], f"{save_test_dir}/{i}.pt")
                elif augment:
                    augmented = augment_image(star_img, mask_ten, prob=1, progress=progress)
                    for k, (img, mask) in enumerate(augmented):
                        img = type_conversion.convert_to_tensor(img, track_meta=False)
                        mask = type_conversion.convert_to_tensor(mask, track_meta=False)
                        torch.save([img, mask], f"{save_train_dir}/{i}_{k}.pt")
                else:
                    torch.save([star_img, mask_ten], f"{save_train_dir}/{i}.pt")

            progress.update(task, advance=1)
            progress.refresh()

    




def augment_prl_images(train_img: Union[torch.Tensor, List[torch.Tensor]], train_label: torch.Tensor, progress: Progress = None) -> Tuple[list, list]:
    
    # find slices with PRLs
    prl_slices = torch.where(train_label[:, 2] > 0)[0]
    prl_slices = torch.unique(prl_slices)

    if isinstance(train_img, torch.Tensor):
        train_img = [train_img]
    
    assert isinstance(train_img, list), "train_img must be a list of tensors or a single tensor!"


    prl_imgs = [img[prl_slices] for img in train_img]
    prl_labels = train_label[prl_slices]

    rand_angle_rad = np.random.uniform(-math.pi / 8, math.pi / 8)

    rotation = Rotate(angle=rand_angle_rad, keep_size=True)  # Use a single angle for 2D data
    flip = Flip(spatial_axis=[0,1])
    gaussion_noise = RandGaussianNoise(prob=1, mean = 0, std=0.1)
    augmentations = [rotation, flip, gaussion_noise]

    augmented_prl_imgs = [[] for _ in range(len(prl_imgs))]
    augmented_prl_labels = []

    if progress is not None:
        task = progress.add_task("Augmenting PRL images...", total=len(prl_imgs[0].shape[0]) * len(augmentations))

    for i in range(prl_imgs[0].shape[0]):
        for j in range(len(augmentations)):
            
            for k, prl_img in enumerate(prl_imgs):
                augmented_prl_imgs[k].append(prl_img[i])

            augmented_prl_labels.append(prl_labels[i])

            for k, prl_img in enumerate(prl_imgs):
                augmented_prl_imgs[k].append(type_conversion.convert_to_tensor(augmentations[j](prl_img[i]), track_meta=False))

            if not isinstance(augmentations[j], RandGaussianNoise):
                augmented_prl_labels.append(type_conversion.convert_to_tensor(augmentations[j](prl_labels[i]), track_meta=False))
            else:
                augmented_prl_labels.append(prl_labels[i])
            if progress is not None:
                progress.update(task, advance=1)
                progress.refresh()
    
    if progress is not None:
        progress.remove_task(task)
        
    return *augmented_prl_imgs, augmented_prl_labels






def augment_image(image: torch.Tensor, mask: torch.Tensor, prob: float = 1.0, progress: Progress = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:

    rand_angle_rad_x = np.random.uniform(-math.pi / 8, math.pi / 8)
    rand_angle_rad_y = np.random.uniform(-math.pi / 8, math.pi / 8)

    rand_scale = np.random.uniform(0.9, 1.1)
    rand_translate = np.random.uniform(-0.1, 0.1)

    rotation = Rotate(angle=[0, rand_angle_rad_x, rand_angle_rad_y], keep_size=True)
    flip = Flip(spatial_axis=[1,2])
    gaussion_noise = RandGaussianNoise(prob=1, mean = 0, std=0.1)
    affine = Affine(rotate_params=[0, rand_angle_rad_x, rand_angle_rad_y], scale_params=[1, rand_scale, rand_scale], translate_params=[0, rand_translate, rand_translate], padding_mode="border")

    transforms = [rotation, flip, gaussion_noise, affine]
    if progress is not None:
        task = progress.add_task("Augmenting data...", total = len(transforms))

    new_images = [(image, mask)]

    for i, transform in enumerate(transforms):
        new_image = transform(image)
        if isinstance(transform, RandGaussianNoise):
            new_mask = mask
        else:
            new_mask = transform(mask)
        
        if (isinstance(new_image, tuple)):
            new_image = new_image[0]
            new_mask = new_mask[0]
        
        new_images.append((new_image, new_mask))
        if progress is not None:
            progress.update(task, advance=1)
            progress.refresh()
    
    if progress is not None:
        progress.remove_task(task)
    
    return new_images



def augment_dataset(images: torch.Tensor, labels = torch.Tensor, progress = None):

    rotation = RandRotate(range_x = 0.4, prob=1, keep_size=True)
    flip = RandFlip(spatial_axis=[1,2], prob=1)
    gaussion_noise = RandGaussianNoise(prob=1, mean = 0, std=0.1)
    affine = RandAffine(prob=1, rotate_range=0.4, scale_range=0.1, translate_range=0.1, padding_mode="border")

    transforms = [rotation, flip, gaussion_noise, affine]

    N = images.shape[0]

    #print((N * (len(transforms) + 1), *images.shape[1:]))

    new_images = torch.empty((N * (len(transforms) + 1), *images.shape[1:]), dtype=images.dtype)
    new_labels = torch.empty((N * (len(transforms) + 1), *labels.shape[1:]), dtype=labels.dtype)

    
    new_images[:N] = images
    new_labels[:N] = labels

    if progress is not None:
        task = progress.add_task("Augmenting data...", total = len(images))

    for i, (image, label) in enumerate(zip(images, labels)):
        for j, transform in enumerate(transforms):
            new_image = transform(image)
            new_images[N + i * len(transforms) + j] = new_image
            new_labels[N + i * len(transforms) + j] = label
        if progress is not None:
            progress.update(task, advance=1)
            progress.refresh()
    
    if progress is not None:
        progress.remove_task(task)

    return new_images, new_labels




def save_lazy_dataset(train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels, save_train_dir, save_val_dir, save_test_dir):
    for path in [save_train_dir, save_val_dir, save_test_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    for i in range(train_imgs.shape[0]):
        torch.save([train_imgs[i], val_imgs[i]], f"{save_train_dir}/{i}.pt")
    for i in range(val_imgs.shape[0]):
        torch.save([val_imgs[i], val_labels[i]], f"{save_val_dir}/{i}.pt")
    for i in range(test_imgs.shape[0]):
        torch.save([test_imgs[i], test_labels[i]], f"{save_test_dir}/{i}.pt")