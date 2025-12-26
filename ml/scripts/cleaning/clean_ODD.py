# ========================================================================================================
# ODD has useful classes of AMD, Glaucoma and Normal. this script just filters the cvs file for these file. it then copies each of the images into /data inside their respective disease folders. 

# Note: the naming convention we have used is {dataset_name}_{label}_{image_name}
# ========================================================================================================

import os
import shutil
import pandas as pd
from src.utils.paths import ODD_RAW_DATA_PATH, PROCESSED_DATA_PATH

def clean_rename_move_ODD(images_source_path, images_destination_path, csv_path, dataset_name):

    df = pd.read_csv(csv_path)

    # filters the csv file
    normal_list = set(df[(df["Normal"]==1)]["ID"].astype(str))
    glaucoma_list = set(df[(df["Glaucoma"]==1)]["ID"].astype(str))
    amd_list = set(df[(df["AMD"]==1)]["ID"].astype(str))

    for image_name in normal_list:
        try:
            shutil.copy2(os.path.join(images_source_path, image_name), os.path.join(images_destination_path, "Normal"))
            shutil.move(os.path.join(images_destination_path,"Normal",image_name), os.path.join(images_destination_path,"Normal", f"{dataset_name}_Normal_{image_name}"))
        
        except FileNotFoundError:
            print("file not found")
            return
        except Exception as e:
            print(f"an error occured: {e}")
            return

    for image_name in amd_list:
        try:
            shutil.copy2(os.path.join(images_source_path, image_name), os.path.join(images_destination_path, "AMD"))
            shutil.move(os.path.join(images_destination_path,"AMD",image_name), os.path.join(images_destination_path,"AMD", f"{dataset_name}_AMD_{image_name}"))
        
        except FileNotFoundError:
            print("file not found")
            return
        except Exception as e:
            print(f"an error occured: {e}")
            return
        
    for image_name in glaucoma_list:
        try:
            shutil.copy2(os.path.join(images_source_path, image_name), os.path.join(images_destination_path, "Glaucoma"))
            shutil.move(os.path.join(images_destination_path,"Glaucoma",image_name), os.path.join(images_destination_path,"Glaucoma", f"{dataset_name}_Glaucoma_{image_name}"))
        
        except FileNotFoundError:
            print("file not found")
            return
        except Exception as e:
            print(f"an error occured: {e}")
            return
    
    print("copy and rename success!")

clean_rename_move_ODD(
    ODD_RAW_DATA_PATH / "Training_Dataset_Final" / "Training_Dataset_Final",
    PROCESSED_DATA_PATH,
    ODD_RAW_DATA_PATH / "Final.csv",
    "ODD"
)