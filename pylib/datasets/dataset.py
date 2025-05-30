import os
import SimpleITK as sitk
import torch
from collections import deque
from torch.utils.data import Dataset, TensorDataset
from torchvision.transforms import Normalize
from torchvision.transforms import functional as VF
from torch.utils.data import random_split
import torch.nn.functional as F
from rich.progress import Progress
from enum import Enum


import pylib.imaging.lesion_tools as lt
import pylib.imaging.transforms as transforms
from typing import List, Type
import gc




def create_lesion_seg_dataset(
        patient_dirs: Type[List[str]],
        flair_fname: str, 
        lesion_fname: str, 
        save_seg_train_path: str,
        save_seg_val_path: str,
        save_seg_test_path: str,
        num_val_samples: int = 1, 
        num_test_samples: int = 1
    ):

    """
    Create a dataset for lesion segmentation from T2 FLAIR and lesion masks.
    Args:
        patient_dirs (List[str]): List of directories containing patient data.
        flair_fname (str): Filename of the FLAIR image.
        lesion_fname (str): Filename of the lesion mask.
        save_seg_train_path (str): Path to save the training dataset.
        save_seg_val_path (str): Path to save the validation dataset.
        save_seg_test_path (str): Path to save the test dataset.
        num_val_samples (int): Number of samples for validation.
        num_test_samples (int): Number of samples for testing.
    Returns:
        None
    """

    
    flair_imgs = []
    lesion_imgs = []

    with Progress() as progress:

        task = progress.add_task("Loading data", total=len(patient_dirs))

        for patient_dir in patient_dirs:

            progress.update(task, description=f"Loading data for patient {patient_dir}...")
            progress.refresh()

            flair_ten, lesions_ten = process_patient(patient_dir, [flair_fname, lesion_fname])

            if flair_ten is not None and lesions_ten is not None:
                flair_imgs.append(flair_ten)
                lesion_imgs.append(lesions_ten)

            progress.update(task, advance=1)
            progress.refresh()
        


        train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = manual_train_val_test_split(flair_imgs, lesion_imgs, num_val_samples, num_test_samples)
        train_imgs, val_imgs, test_imgs = normalize_imgs(train_imgs, val_imgs, test_imgs)
        train_dataset, val_dataset, test_dataset = create_sub_dataset(train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels)

        torch.save(train_dataset, save_seg_train_path)
        torch.save(val_dataset, save_seg_val_path)
        torch.save(test_dataset, save_seg_test_path)
    




def create_prl_class_dataset_2d(
        patient_dirs: Type[List[str]],
        phase_fname: str,
        lesion_fname: str,
        prl_fname: str,
        save_class_train_path: str,
        save_class_val_path: str,
        save_class_test_path: str,
        train_val_test_split: list,
        img_size: int,
        lesion_expansion: int,
        min_inter_lesion_distance: int,
        mask: bool
    ):

    """
    Create a dataset for PRL classification from T2* phase images, lesion masks, and PRL masks.
    Args:
        patient_dirs (List[str]): List of directories containing patient data.
        phase_fname (str): Filename of the T2* phase image.
        lesion_fname (str): Filename of the lesion mask.
        prl_fname (str): Filename of the PRL mask.
        save_class_train_path (str): Path to save the training dataset.
        save_class_val_path (str): Path to save the validation dataset.
        save_class_test_path (str): Path to save the test dataset.
        train_val_test_split (list): List containing the proportions for train, validation, and test datasets.
        img_size (int): Size of the sub-images to be extracted.
        lesion_expansion (int): Expansion factor for the lesions.
        min_inter_lesion_distance (int): Minimum distance between lesions.
        mask (bool): Whether to apply a mask to the images.
    Returns:
        None
    """


    phase_imgs = []
    lesion_imgs = []
    prl_imgs = []
    
    with Progress() as progress:

        task = progress.add_task("Loading data", total=len(patient_dirs))

        for patient_dir in patient_dirs:

            progress.update(task, description=f"Loading data for patient {patient_dir}...")
            progress.refresh()

            phase_ten, lesions_ten, prl_ten = process_patient(patient_dir, [phase_fname, lesion_fname, prl_fname])

            if phase_ten is not None and lesions_ten is not None and prl_ten is not None:
                phase_imgs.append(phase_ten)
                lesion_imgs.append(lesions_ten)
                prl_imgs.append(prl_ten)

            progress.update(task, advance=1)
            progress.refresh()
        
        img_ten, lesions_ten, prl_ten = torch.cat(phase_imgs), torch.cat(lesion_imgs), torch.cat(prl_imgs)
        img_ten, lesions_ten, prl_ten = img_ten.flatten(0, 1), lesions_ten.flatten(0, 1), prl_ten.flatten(0, 1)

        lesion_imgs, lesion_labels = process_prl_tensors_to_sub_images_2d(img_ten, lesions_ten, prl_ten, img_size, lesion_expansion, min_inter_lesion_distance, mask, progress)

        prl_dataset = TensorDataset(lesion_imgs, lesion_labels)

        train_dataset, val_dataset, test_dataset = random_split(prl_dataset, train_val_test_split)

        normalizer = Normalize(torch.mean(train_dataset[:][0]), torch.std(train_dataset[:][0]))
        train_dataset = TensorDataset(normalizer(train_dataset[:][0]), train_dataset[:][1])
        val_dataset = TensorDataset(normalizer(val_dataset[:][0]), val_dataset[:][1])
        test_dataset = TensorDataset(normalizer(test_dataset[:][0]), test_dataset[:][1])

        torch.save(train_dataset, save_class_train_path)
        torch.save(val_dataset, save_class_val_path)
        torch.save(test_dataset, save_class_test_path)



def create_prl_class_dataset_3d(
        patient_dirs: Type[List[str]],
        phase_fname: str,
        lesion_fname: str,
        prl_fname: str,
        img_size: tuple,
        lesion_expansion: int,
        min_inter_lesion_distance: int,
        mask: bool,
        save_class_train_path: str,
        save_class_val_path: str,
        save_class_test_path: str,
        num_val_samples: int = 1, 
        num_test_samples: int = 1,
    ):    

    """
    3D version of create_prl_class_dataset_2d.
    """

    imgs_3d = []
    labels_3d = []

                        
    with Progress() as progress:

        task = progress.add_task("Loading data", total=len(patient_dirs))

        for i, patient_dir in enumerate(patient_dirs):

            progress.update(task, description=f"Loading data for patient {patient_dir}...")
            progress.refresh()

            phase_ten, lesions_ten, prl_ten = process_patient(patient_dir, [phase_fname, lesion_fname, prl_fname], add_channel_dim=False)

            if phase_ten is not None and lesions_ten is not None and prl_ten is not None:
                lesion_imgs, lesion_labels = process_prl_tensors_to_sub_images_3d(phase_ten, lesions_ten, prl_ten, img_size, lesion_expansion, min_inter_lesion_distance, mask, progress)
                
                imgs_3d.append(lesion_imgs)
                labels_3d.append(lesion_labels)

                progress.update(task, advance=1)
                progress.refresh() 

        print("Splitting data...")
        train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = manual_train_val_test_split(imgs_3d, labels_3d, num_val_samples, num_test_samples, cat = False)

        #combine patients
        print("Combining patients...")
        train_imgs, val_imgs, test_imgs = combine_patients_3d(train_imgs, val_imgs, test_imgs)
        train_labels, val_labels, test_labels = combine_patients_3d(train_labels, val_labels, test_labels)

        print("Normalizing images...")
        train_imgs, val_imgs, test_imgs = normalize_imgs_3d(train_imgs, val_imgs, test_imgs)

        print("Stacking images...")
        train_imgs = torch.stack(train_imgs)
        val_imgs = torch.stack(val_imgs)
        test_imgs = torch.stack(test_imgs)

        train_labels = torch.stack(train_labels)
        val_labels = torch.stack(val_labels)
        test_labels = torch.stack(test_labels)

        train_dataset, val_dataset, test_dataset = create_sub_dataset(train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels)

        print("Saving datasets...")
        torch.save(train_dataset, save_class_train_path)
        torch.save(val_dataset, save_class_val_path)
        torch.save(test_dataset, save_class_test_path)

        print("Done!")



def process_prl_tensors_to_sub_images_2d(img_ten: torch.Tensor, lesions_ten: torch.Tensor, prl_ten: torch.Tensor, img_size: int, lesion_expansion: int, min_inter_lesion_distance: int, mask: bool = True, progress: Progress = None):
    """
    Process tensors to extract sub-images for PRL classification.
    Args:
        img_ten (torch.Tensor): Tensor containing the T2* phase images.
        lesions_ten (torch.Tensor): Tensor containing the lesion masks.
        prl_ten (torch.Tensor): Tensor containing the PRL masks.
        img_size (int): Size of the sub-images to be extracted.
        lesion_expansion (int): Expansion factor for the lesions.
        min_inter_lesion_distance (int): Minimum distance between lesions.
        mask (bool): Whether to apply a mask to the images.
    Returns:
        Tuple[list, list]: A tuple containing the processed images and labels.
    """

    #Grow PRLs to fit to the expanded lesions
    if progress is not None:
        task = progress.add_task("Growing PRLs", total=1)

    lesions_ten, prl_ten = lt.grow_all_regions(lesions_ten, prl_ten)

    
    if progress is not None:
        progress.update(task, advance=1)
        progress.refresh()

    #Assign class labels to PRLs
    lesions_ten[prl_ten == 1] = 2

    lesion_imgs = []
    lesion_labels = []

    if progress is not None:
        task = progress.add_task("Splitting images by lesion", total=lesions_ten.shape[0])

    assert(len(lesions_ten.shape) == 3 and len(img_ten.shape) == 3 and len(prl_ten.shape) == 3)

    for i in range(lesions_ten.shape[0]):
        imgs, labels = lt.split_img_by_lesion(img_ten[i], lesions_ten[i], img_size, expand = lesion_expansion, min_inter_lesion_distance = min_inter_lesion_distance, mask = mask)
        lesion_imgs.extend(imgs)
        lesion_labels.extend(labels)

        if progress is not None:
            progress.update(task, advance=1)
            progress.refresh()
    
    #Stack and add channel dimension to images
    lesion_imgs = torch.stack(lesion_imgs).unsqueeze(1)
    lesion_labels = torch.stack(lesion_labels).unsqueeze(1)

    #Set PRLs to class 1 (positive) and other lesions to class 0 (negative)
    lesion_labels[lesion_labels == 1] = 0
    lesion_labels[lesion_labels == 2] = 1

    return lesion_imgs, lesion_labels

        
        


def manual_train_val_test_split(imgs: list, labels: list, num_val_samples: int, num_test_samples: int, cat: bool = True):
    """
    Manually split the dataset into training, validation, and test sets.
    Args:
        imgs (list): List of image tensors.
        labels (list): List of label tensors.
        num_val_samples (int): Number of validation samples.
        num_test_samples (int): Number of test samples.
        cat (bool): Whether to concatenate the images and labels.
    Returns:
        Tuple: A tuple containing the training, validation, and test sets.
    """
    train_imgs = imgs[:-num_val_samples - num_test_samples]
    val_imgs = imgs[-num_val_samples - num_test_samples: -num_test_samples]
    test_imgs = imgs[-num_test_samples:]

    train_labels = labels[:-num_val_samples - num_test_samples]
    val_labels = labels[-num_val_samples - num_test_samples: -num_test_samples]
    test_labels = labels[-num_test_samples:]

    if cat:
        train_imgs = torch.cat(train_imgs)
        val_imgs = torch.cat(val_imgs)
        test_imgs = torch.cat(test_imgs)

        train_labels = torch.cat(train_labels)
        val_labels = torch.cat(val_labels)
        test_labels = torch.cat(test_labels)

    return train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels



def normalize_imgs(train_imgs: torch.Tensor, val_imgs: torch.Tensor, test_imgs: torch.Tensor):
    """
    Normalize the images using the mean and standard deviation of the training set.
    Args:
        train_imgs (torch.Tensor): Training images tensor.
        val_imgs (torch.Tensor): Validation images tensor.
        test_imgs (torch.Tensor): Test images tensor.
    Returns:
        Tuple: A tuple containing the normalized training, validation, and test images. 
    """
    normalizer = Normalize(torch.mean(train_imgs), torch.std(train_imgs))
    train_imgs = normalizer(train_imgs)
    val_imgs = normalizer(val_imgs)
    test_imgs = normalizer(test_imgs)
    return train_imgs, val_imgs, test_imgs

def normalize_imgs_3d(train_imgs: list, val_imgs: list, test_imgs: list):
    """
    Normalize the 3D images using the mean and standard deviation of the training set.
    Args:
        train_imgs (list): List of training images tensors.
        val_imgs (list): List of validation images tensors.
        test_imgs (list): List of test images tensors.
    Returns:
        Tuple: A tuple containing the normalized training, validation, and test images.
    """
    mean = 0
    std = 0
    for img in train_imgs:
        mean += torch.mean(img)
        std += torch.std(img)
    mean /= len(train_imgs)
    std /= len(train_imgs)
    normalizer = Normalize(mean, std)
    train_imgs = [normalizer(img) for img in train_imgs]
    val_imgs = [normalizer(img) for img in val_imgs]
    test_imgs = [normalizer(img) for img in test_imgs]
    return train_imgs, val_imgs, test_imgs

def create_sub_dataset(train_imgs: torch.Tensor, train_labels: torch.Tensor, val_imgs: torch.Tensor, val_labels: torch.Tensor, test_imgs: torch.Tensor, test_labels: torch.Tensor):
    """
    Create a TensorDataset for training, validation, and test sets.
    Args:
        train_imgs (torch.Tensor): Training images tensor.
        train_labels (torch.Tensor): Training labels tensor.
        val_imgs (torch.Tensor): Validation images tensor.
        val_labels (torch.Tensor): Validation labels tensor.
        test_imgs (torch.Tensor): Test images tensor.
        test_labels (torch.Tensor): Test labels tensor.
    Returns:
        Tuple: A tuple containing the training, validation, and test datasets.
    """
    train_dataset = TensorDataset(train_imgs, train_labels)
    val_dataset = TensorDataset(val_imgs, val_labels)
    test_dataset = TensorDataset(test_imgs, test_labels)
    return train_dataset, val_dataset, test_dataset



def process_patient(patient_dir: str, fnames: list, add_channel_dim: bool = True):
    """
    Process a patient's data by loading the specified nifti files and applying transformations.
    Args:
        patient_dir (str): Directory containing the patient's nifti files.
        fnames (list): List of nifti file names to process.
        add_channel_dim (bool): Whether to add a channel dimension to the tensors.
    Returns:
        list: A list of processed image tensors.
    """
    if not os.path.isdir(patient_dir):
        print(f"The directory: {patient_dir} does not exist.")
        return None
    
    processed_tensors = []

    for fname in fnames:
        if not fname.endswith("nii.gz"):
            print(f"The file: {fname} is not a nifti file.")
            return None

        path = os.path.join(patient_dir, fname)

        if not os.path.isfile(path):
            print(f"The file: {path} does not exist in {patient_dir}.")
            return None
        
        ten = transforms.load_nifti_to_tensors(path)
        ten = transforms.crop_tensor(ten)
        ten = VF.rotate(ten, 90)

        if add_channel_dim:
            # Add channel dimension
            ten = ten.unsqueeze(0)

        processed_tensors.append(ten)

    return processed_tensors




def process_prl_tensors_to_sub_images_3d(phase_imgs: torch.Tensor, lesion_imgs: torch.Tensor, prl_imgs: torch.Tensor, img_size: tuple, lesion_expansion: int, min_inter_lesion_distance: int, mask: bool = True, progress: Progress = None):
    """
    Process the PRL tensors to extract sub-images.
    Args:
        phase_imgs (torch.Tensor): Tensor containing the T2* phase images.
        lesion_imgs (torch.Tensor): Tensor containing the lesion masks.
        prl_imgs (torch.Tensor): Tensor containing the PRL masks.
        img_size (int): Size of the sub-images to be extracted.
        lesion_expansion (int): Expansion factor for the lesions.
        min_inter_lesion_distance (int): Minimum distance between lesions.
        mask (bool): Whether to apply a mask to the images.
        progress (Progress, optional): Progress tracking object.
    Returns:
        Tuple[list, list]: A tuple containing the processed images and labels.
    """

    assert(len(phase_imgs) == len(lesion_imgs) == len(prl_imgs))
    assert(len(phase_imgs.shape) == 3 and len(lesion_imgs.shape) == 3 and len(prl_imgs.shape) == 3)
    assert(len(img_size) == 3)

    lesion_imgs, prl_imgs = lt.grow_all_regions(lesion_imgs, prl_imgs)
    
    #Assign class labels to PRLs
    for i in range(len(lesion_imgs)):
        lesion_imgs[i][prl_imgs[i] == 1] = 2


    imgs, labels = lt.split_img_by_lesions_3d(
        phase_imgs, lesion_imgs, img_size, 
        expand=lesion_expansion, 
        min_inter_lesion_distance=min_inter_lesion_distance, 
        mask=mask,
        progress=progress
    )

    for i in range(len(imgs)):
        labels[i][labels[i] == 1] = 0
        labels[i][labels[i] == 2] = 1

    return imgs, labels

def combine_patients_3d(train_imgs: list, val_imgs: list, test_imgs: list):
    train_imgs = [i for sublist in train_imgs for i in sublist]
    val_imgs = [i for sublist in val_imgs for i in sublist]
    test_imgs = [i for sublist in test_imgs for i in sublist]
    return train_imgs, val_imgs, test_imgs

