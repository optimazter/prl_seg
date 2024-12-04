import pylib.nifti as nii
import config as c

def main():
    nii.create_lesion_dataset(
        load_dir=c.ORIG_PATH, 
        training_data_path=c.PATH_TRAINING_DATASET, 
        validation_data_path=c.PATH_VALIDATION_DATASET,
        test_data_path=c.PATH_TEST_DATASET,
        img_path = c.PHASE, 
        all_lesions_path=c.SEG_2_PHASE, 
        prl_path=c.PHASE_UNWRAPPED_UBMASK,
        train_val_test_split=[c.TRAINING_SPLIT, c.VALIDATION_SPLIT, c.TEST_SPLIT]
    )


if __name__ == "__main__":
    main()