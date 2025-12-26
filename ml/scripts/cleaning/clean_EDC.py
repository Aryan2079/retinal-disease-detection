# ===================================================================================================
# this script deletes the cataract folder and then moves the remaining folders to the /data folder following their respective naming convention
# ===================================================================================================

import shutil
import os
from move_images_and_rename import move_images_and_rename
from src.utils.paths import DR_PROCESSED_PATH, NORMAL_PROCESSED_PATH, GLAUCOMA_PROCESSED_PATH, EDC_RAW_DATA_PATH

def clean_EDC(directory_to_delete):
    if os.path.exists(directory_to_delete):
        try:
            shutil.rmtree(directory_to_delete)
            print(f"Directory '{directory_to_delete}' and all its contents have been deleted.")

        except Exception as e:
            print(f"Error deleting directory '{directory_to_delete}': {e}")
    else:
        print(f"Directory cant be deleted. {directory_to_delete} doesnt exist")


# cleaning EDC dataset
clean_EDC(EDC_RAW_DATA_PATH / "cataract")


# for dr
move_images_and_rename(EDC_RAW_DATA_PATH / "diabetic_retinopathy",
                       DR_PROCESSED_PATH,
                       "EDC")

# for normal
move_images_and_rename(EDC_RAW_DATA_PATH / "normal",
                       NORMAL_PROCESSED_PATH,
                       "EDC")

#for glaucoma
move_images_and_rename(EDC_RAW_DATA_PATH / "glaucoma",
                       GLAUCOMA_PROCESSED_PATH,
                        "EDC")