# ===================================================================================================
# this script deletes all the folders except "Images" in "G1020" and "origa" dataset and moves them into /data following the naming convention
# ===================================================================================================

import pandas as pd
import os
import shutil

def clean_GID(directory_to_clean):
    dir_list = os.listdir(directory_to_clean)

    for dir in dir_list:
        if dir == "Images" or dir.endswith(".csv"):
            continue

        if os.path.exists(os.path.join(directory_to_clean, dir)):
            try:
                shutil.rmtree(os.path.join(directory_to_clean, dir))
                print(f"Directory '{os.path.join(directory_to_clean, dir)}' and all its contents have been deleted.")

            except Exception as e:
                print(f"Error deleting directory '{os.path.join(directory_to_clean, dir)}': {e}")
        else:
            print(f"Directory cant be deleted. {os.path.join(directory_to_clean, dir)} doesnt exist")

def move_and_rename_csv_g1020(image_source_path, image_destination_path, csv_path, dataset_name):
    df = pd.read_csv(csv_path)

    glaucoma_list = set(df[(df["binaryLabels"]==1)]["imageID"].astype(str))
    normal_list = set(df[(df["binaryLabels"]==0)]["imageID"].astype(str))

    for image_name in glaucoma_list:
        try:
            shutil.copy(os.path.join(image_source_path, image_name), os.path.join(image_destination_path,"Glaucoma"))
            shutil.move(os.path.join(image_destination_path, "Glaucoma", image_name), os.path.join(image_destination_path, "Glaucoma", f"{dataset_name}_Glaucoma_{image_name}"))
        except FileNotFoundError:
            print("file not found error")
            return
        except Exception as e:
            print(f"an exception occured: {e}")
            return
        
    for image_name in normal_list:
        try:
            shutil.copy(os.path.join(image_source_path, image_name), os.path.join(image_destination_path,"Normal"))
            shutil.move(os.path.join(image_destination_path, "Normal", image_name), os.path.join(image_destination_path, "Normal", f"{dataset_name}_Normal_{image_name}"))
        except FileNotFoundError:
            print("file not found error")
            return
        except Exception as e:
            print(f"an exception occured: {e}")
            return
    
    print("move and rename success g1020!!")


def move_and_rename_csv_origa(image_source_path, image_destination_path, csv_path, dataset_name):

    df = pd.read_csv(csv_path)

    glaucoma_list = set(df[(df["Glaucoma"]==1)]["Filename"].astype(str))
    normal_list = set(df[(df["Glaucoma"]==0)]["Filename"].astype(str))

    for image_name in glaucoma_list:
        try:
            shutil.copy(os.path.join(image_source_path, image_name), os.path.join(image_destination_path,"Glaucoma"))
            shutil.move(os.path.join(image_destination_path, "Glaucoma", image_name), os.path.join(image_destination_path, "Glaucoma", f"{dataset_name}_Glaucoma_{image_name}"))
        except FileNotFoundError:
            print("file not found error")
            return
        except Exception as e:
            print(f"an exception occured: {e}")
            return
        
    for image_name in normal_list:
        try:
            shutil.copy(os.path.join(image_source_path, image_name), os.path.join(image_destination_path,"Normal"))
            shutil.move(os.path.join(image_destination_path, "Normal", image_name), os.path.join(image_destination_path, "Normal", f"{dataset_name}_Normal_{image_name}"))
        except FileNotFoundError:
            print("file not found error")
            return
        except Exception as e:
            print(f"an exception occured: {e}")
            return
    
    print("move and rename success origa!!")

# cleaning

# for g1020
clean_GID(r"C:\Users\aryan\Projects\Major\raw\Glaucoma_Fundus_Imaging_Dataset\Dataset\G1020") 

# for origa
clean_GID(r"C:\Users\aryan\Projects\Major\raw\Glaucoma_Fundus_Imaging_Dataset\Dataset\ORIGA")


# move and rename

# for g1020
move_and_rename_csv_g1020(
    r"C:\Users\aryan\Projects\Major\raw\Glaucoma_Fundus_Imaging_Dataset\Dataset\G1020\Images",
    r"C:\Users\aryan\Projects\Major\data\images",
    r"C:\Users\aryan\Projects\Major\raw\Glaucoma_Fundus_Imaging_Dataset\Dataset\G1020\G1020.csv",
    "GID"
)

# for origa
move_and_rename_csv_origa(
    r"C:\Users\aryan\Projects\Major\raw\Glaucoma_Fundus_Imaging_Dataset\Dataset\ORIGA\Images",
    r"C:\Users\aryan\Projects\Major\data\images",
    r"C:\Users\aryan\Projects\Major\raw\Glaucoma_Fundus_Imaging_Dataset\Dataset\ORIGA\OrigaList.csv",
    "GID"
)

        