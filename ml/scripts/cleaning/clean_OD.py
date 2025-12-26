# =======================================================================================================
# Note: we havent added the data for DR from this dataset so that it doesnt increase the data imbalance even more
# this script just selects the required labels and moves them in the /data folder
# =======================================================================================================

from move_images_and_rename import move_images_and_rename
import os
import shutil
from src.utils.paths import OD_RAW_DATA_PATH, PROCESSED_DATA_PATH

splits = ["A", "G", "N"]

def clean_OD(directory_to_clean):
    dir_list = os.listdir(directory_to_clean)

    for dir in dir_list:
        if dir in splits or dir.endswith(".csv"):
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
clean_OD(OD_RAW_DATA_PATH)

# moving
for split in splits:
    folder_label = None

    if split == "A":
        folder_label = "AMD"
    elif split == "G":
        folder_label = "Glaucoma"
    elif split == "N":
        folder_label = "Normal"

    move_images_and_rename(
        OD_RAW_DATA_PATH / split,
        PROCESSED_DATA_PATH / folder_label,
        "OD"
    )

print("OD data transfer complete")
