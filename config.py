import torch

#Loaded files
ORIG_PATH = "assets/Lou_et_al_dataset"
FLAIR = "flair_pp.nii.gz"
PHASE = "phase_pp.nii.gz"
PHASE_UNWRAPPED_UBMASK = "phase_unwrapped_ubmask.nii.gz"
SEG_2_PHASE = "seg2phase.nii.gz"
T1 = "t1_pp.nii.gz"

#Generated files
PATH_TRAINING_DATASET = "output/train_dataset.pt"
PATH_VALIDATION_DATASET = "output/val_dataset.pt"
PATH_TEST_DATASET = "output/test_dataset.pt"

PATH_RAW_DATASET = "output/dataset.pt"

#Training parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
BATCH_SIZE = 8
LEARNING_RATE = 0.001
SEED = 24
EPOCHS = 2
TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]

#Image parameters
CHANNEL_PHASE = 0
CHANNEL_FLAIR = 1