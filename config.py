import torch

#Loaded files
ORIG_PATH = "assets/Lou_et_al_dataset"
FLAIR = "flair_pp.nii.gz"
PHASE = "phase_pp.nii.gz"
PHASE_UNWRAPPED_UBMASK = "phase_unwrapped_ubmask.nii.gz"
SEG_2_PHASE = "seg2phase.nii.gz"
T1 = "t1_pp.nii.gz"

#Generated files
PATH_VISUALIZATION_DATASET = "output/vis_dataset.pt"


PATH_TRAIN_LESION_SEG_DATASET = "output/train_lesion_seg_dataset.pt"
PATH_VAL_LESION_SEG_DATASET = "output/val_lesion_seg_dataset.pt"
PATH_TEST_LESION_SEG_DATASET = "output/test_lesion_dataset.pt"

PATH_2D_TRAIN_PRL_CLASS_DATASET = "output/train_prl_class_dataset.pt"
PATH_2D_VAL_PRL_CLASS_DATASET = "output/val_prl_class_dataset.pt"
PATH_2D_TEST_PRL_CLASS_DATASET = "output/test_prl_class_dataset.pt"


PATH_3D_TRAIN_PRL_CLASS_DATASET = "output/train_prl_class_dataset_3d.pt"
PATH_3D_VAL_PRL_CLASS_DATASET = "output/val_prl_class_dataset_3d.pt"
PATH_3D_TEST_PRL_CLASS_DATASET = "output/test_prl_class_dataset_3d.pt"


PATH_TRAIN_SWI_DATASET = "output/train_swi_dataset.pt"
PATH_VAL_SWI_DATASET = "output/val_swi_dataset.pt"
PATH_TEST_SWI_DATASET = "output/test_swi_dataset.pt"

DIR_TRAIN_SWI_DATASET = "output/train_swi_dataset"
DIR_VAL_SWI_DATASET =     "output/val_swi_dataset"
DIR_TEST_SWI_DATASET =   "output/test_swi_dataset"


DIR_TRAIN_SWI_DATASET_3D = "output/train_swi_dataset_3d"
DIR_VAL_SWI_DATASET_3D = "output/val_swi_dataset_3d"
DIR_TEST_SWI_DATASET_3D = "output/test_swi_dataset_3d"


OUT_DIR = "output"
PATH_LOG = "output/log.csv"

#Training parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]
SEED = 24





#Image parameters
CHANNEL_PHASE = 0
CHANNEL_FLAIR = 1
SUB_LESION_IMG_SIZE = (64, 64)
SUB_LESION_EXPANSION = 25
MIN_SUB_LESION_DISTANCE = 10