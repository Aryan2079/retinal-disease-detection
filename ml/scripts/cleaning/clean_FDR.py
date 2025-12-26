# =======================================================================================================
# this dataset has dr fundus images in 5 severity. 0 severity is normal fundus images. this script saves 0 severity into /data/images/normal and the rest in /data/images/dr
# =======================================================================================================

from move_images_and_rename import move_images_and_rename
from src.utils.paths import FDR_RAW_DATA_PATH, NORMAL_PROCESSED_PATH, DR_PROCESSED_PATH

data_splits = ["test","train","val"]

for split in data_splits:
    for i in range(5):
        if i == 0:
            move_images_and_rename(FDR_RAW_DATA_PATH / split / str(i),
                                   NORMAL_PROCESSED_PATH,
                                   "FDR")

        move_images_and_rename(FDR_RAW_DATA_PATH / split / str(i),
                               DR_PROCESSED_PATH,
                               "FDR")
