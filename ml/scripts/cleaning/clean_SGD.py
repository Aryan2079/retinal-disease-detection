# =======================================================================================================
# this script deletes all the folders and just keeps the full "fundus folder". it then renames and copies the images into the /data accordingly.
# =======================================================================================================

import os
import pandas as pd
import shutil

def clean_SGD(directory_to_clean):
    dir_list = os.listdir(directory_to_clean)

    for dir in dir_list:
        if dir == "full-fundus" or dir.endswith(".csv"):
            continue

        if os.path.exists(os.path.join(directory_to_clean, dir)):
            try:
                shutil.rmtree(os.path.join(directory_to_clean, dir))
                print(f"Directory '{os.path.join(directory_to_clean, dir)}' and all its contents have been deleted.")

            except Exception as e:
                print(f"Error deleting directory '{os.path.join(directory_to_clean, dir)}': {e}")
        else:
            print(f"Directory cant be deleted. {os.path.join(directory_to_clean, dir)} doesnt exist")


def move_and_rename_csv(image_source_path, image_destination_path, csv_path, dataset_name):
    df = pd.read_csv(csv_path)

    normal_list = set(df[(df["types"]==0)]["names"].astype(str) + ".png")
    glaucoma_list = set(df[(df["types"]==1)]["names"].astype(str) + ".png")

    for image_name in normal_list:
        try:
            shutil.copy2(os.path.join(image_source_path, image_name), os.path.join(image_destination_path, "Normal"))
            shutil.move(os.path.join(image_destination_path,"Normal",image_name), os.path.join(image_destination_path, "Normal", f"{dataset_name}_Normal_{image_name}"))
        except FileNotFoundError:
            print("file not found error")
            return 
        except Exception as e:
            print(f"exception occured: {e}")
            return

    for image_name in glaucoma_list:
        try:
            shutil.copy2(os.path.join(image_source_path, image_name), os.path.join(image_destination_path, "Glaucoma"))
            shutil.move(os.path.join(image_destination_path,"Glaucoma",image_name), os.path.join(image_destination_path, "Glaucoma", f"{dataset_name}_Glaucoma_{image_name}"))
        except FileNotFoundError:
            print("file not found error")
            return
        except Exception as e:
            print(f"an exception occured: {e}")
            return
    
    print("move and rename success!!")

#cleaning the dataset
clean_SGD(r"C:\Users\aryan\Projects\Major\raw\Standarized_Glaucoma_Dataset\Dataset")

#for full-fundus images
move_and_rename_csv(
    r"C:\Users\aryan\Projects\Major\raw\Standarized_Glaucoma_Dataset\Dataset\full-fundus\full-fundus",
    r"C:\Users\aryan\Projects\Major\data\images",
    r"C:\Users\aryan\Projects\Major\raw\Standarized_Glaucoma_Dataset\Dataset\metadata - standardized.csv",
    "SGD")