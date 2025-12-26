from move_images_and_rename import move_images_and_rename
import os
from src.utils.paths import AMD_PROCESSED_PATH, NORMAL_PROCESSED_PATH, MDG_RAW_DATA_PATH


# for train amd
move_images_and_rename(
    os.path.join(MDG_RAW_DATA_PATH / "train" / "amd"),
    AMD_PROCESSED_PATH,
    "MDG"
)
# for train normal
move_images_and_rename(
    os.path.join(MDG_RAW_DATA_PATH / "train" / "normal"),
    NORMAL_PROCESSED_PATH,
    "MDG"
)

# for valid amd
move_images_and_rename(
    os.path.join(MDG_RAW_DATA_PATH / "valid" / "amd"),
    AMD_PROCESSED_PATH,
    "MDG"
)
# for valid normal
move_images_and_rename(
    os.path.join(MDG_RAW_DATA_PATH / "valid" / "normal"),
    NORMAL_PROCESSED_PATH,
    "MDG"
)

print("MDG data transfer complete")
