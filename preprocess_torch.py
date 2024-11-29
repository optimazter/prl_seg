import pylib.nifti as nii


ORIG_PATH = "assets/Lou_et_al_dataset"
FLAIR = "flair_pp.nii.gz"
PHASE = "phase_pp.nii.gz"
PHASE_UNWRAPPED_UBMASK = "phase_unwrapped_ubmask.nii.gz"
SEG_2_PHASE = "seg2phase.nii.gz"
T1 = "t1_pp.nii.gz"
DATASET = "output/dataset.pt"


def main():
    nii.create_lesion_dataset(ORIG_PATH, DATASET, PHASE, SEG_2_PHASE, PHASE_UNWRAPPED_UBMASK)


if __name__ == "__main__":
    main()