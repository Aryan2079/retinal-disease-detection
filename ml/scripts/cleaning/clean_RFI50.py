
# =======================================================================================================
# Note: we havent used the data for DR from this dataset so as to not increase the data imbalance even more
# this script just selects the required labels and moves them in the /data folder
# =======================================================================================================

from move_images_and_rename import move_images_and_rename
import os
import shutil
from src.utils.paths import RFI50_RAW_DATA_PATH, AMD_PROCESSED_PATH, GLAUCOMA_PROCESSED_PATH, NORMAL_PROCESSED_PATH

do_not_delete_list = ["AMD", "Glaucoma", "Normal_Fundus"]

def clean_RF150(directory_to_clean):
    dir_list = os.listdir(directory_to_clean)

    for dir in dir_list:
        if dir in do_not_delete_list or dir.endswith(".csv"):
            continue

        if os.path.exists(os.path.join(directory_to_clean, dir)):
            try:
                shutil.rmtree(os.path.join(directory_to_clean, dir))
                print(f"Directory '{os.path.join(directory_to_clean, dir)}' and all its contents have been deleted.")

            except Exception as e:
                print(f"Error deleting directory '{os.path.join(directory_to_clean, dir)}': {e}")
        else:
            print(f"Directory cant be deleted. {os.path.join(directory_to_clean, dir)} doesnt exist")

# cleaning

# for test
clean_RF150(RFI50_RAW_DATA_PATH / "test")
# for train
clean_RF150(RFI50_RAW_DATA_PATH / "train")
# for val
clean_RF150(RFI50_RAW_DATA_PATH / "val")

# moving
splits = ["train", "val", "test"]

for split in splits:
    # for amd
    move_images_and_rename(
        RFI50_RAW_DATA_PATH / split / "AMD",
        AMD_PROCESSED_PATH,
        "RFI50"
    )

    # for glaucoma
    move_images_and_rename(
        RFI50_RAW_DATA_PATH / split / "Glaucoma",
        GLAUCOMA_PROCESSED_PATH,
        "RFI50"
    )

    # for normal
    move_images_and_rename(
        RFI50_RAW_DATA_PATH / split / "Normal_Fundus",
        NORMAL_PROCESSED_PATH,
        "RFI50"
    )

print("RFI50 data transfer complete")
