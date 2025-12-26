# =======================================================================================================
# this dataset has dr fundus images in 5 severity. 0 severity is normal fundus images. this script saves 0 severity into /data/images/normal and the rest in /data/images/dr
# =======================================================================================================

from move_images_and_rename import move_images_and_rename
import os

destination_path_dr = r"C:\Users\aryan\Projects\Major\data\images\DR"
destination_path_normal = r"C:\Users\aryan\Projects\Major\data\images\Normal"
source_path = r"C:\Users\aryan\Projects\Major\raw\Fundus_DR\Dataset\split_dataset"

data_splits = ["test","train","val"]

for split in data_splits:
    for i in range(5):
        if i == 0:
            move_images_and_rename(os.path.join(source_path, split, str(i)), destination_path_normal, "FDR")

        move_images_and_rename(os.path.join(source_path, split, str(i)), destination_path_dr, "FDR")
