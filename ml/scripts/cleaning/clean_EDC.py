# ===================================================================================================
# this script deletes the cataract folder and then moves the remaining folders to the /data folder following their respective naming convention
# ===================================================================================================

import shutil
import os
from move_images_and_rename import move_images_and_rename

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
clean_EDC(r"C:\Users\aryan\Projects\Major\raw\Eye_Disease_Classification\Dataset\cataract")


# for dr
move_images_and_rename(r"C:\Users\aryan\Projects\Major\raw\Eye_Disease_Classification\Dataset\diabetic_retinopathy", r"C:\Users\aryan\Projects\Major\data\images\DR","EDC")

# for normal
move_images_and_rename(r"C:\Users\aryan\Projects\Major\raw\Eye_Disease_Classification\Dataset\normal", r"C:\Users\aryan\Projects\Major\data\images\Normal","EDC")

#for glaucoma
move_images_and_rename(r"C:\Users\aryan\Projects\Major\raw\Eye_Disease_Classification\Dataset\glaucoma", r"C:\Users\aryan\Projects\Major\data\images\Glaucoma","EDC")