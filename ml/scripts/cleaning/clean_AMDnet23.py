from move_images_and_rename import move_images_and_rename
import os
import shutil
from src.utils.paths import AMD_PROCESSED_PATH, NORMAL_PROCESSED_PATH, AMDnet23_RAW_DATA_PATH

def clean_AMDnet23(directory_to_clean):
    dir_list = os.listdir(directory_to_clean)

    for dir in dir_list:
        if dir == "amd" or dir == "normal" or dir.endswith(".csv"):
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

# for train
clean_AMDnet23(AMDnet23_RAW_DATA_PATH / "train")

# for valid
clean_AMDnet23(AMDnet23_RAW_DATA_PATH / "valid")


# moving

# for train amd
move_images_and_rename(
    AMDnet23_RAW_DATA_PATH / "train" / "amd",
    AMD_PROCESSED_PATH,
    "AMDnet23"
)
# for train normal
move_images_and_rename(
    AMDnet23_RAW_DATA_PATH / "train" / "normal",
    NORMAL_PROCESSED_PATH,
    "AMDnet23"
)

# for valid amd
move_images_and_rename(
    AMDnet23_RAW_DATA_PATH / "valid" / "amd",
    AMD_PROCESSED_PATH,
    "AMDnet23"
)
# for valid normal
move_images_and_rename(
    AMDnet23_RAW_DATA_PATH / "valid" / "normal",
    NORMAL_PROCESSED_PATH,
    "AMDnet23"
)

print("AMDnet23 data transfer complete")
