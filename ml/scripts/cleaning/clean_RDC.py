#  ====================================================================================================
#  this script cleans takes the path of the csv file and image folder of the RDC dataset. it simply extracts all the files with DR and AMD. rewrites the original csv with the filtered data and removes all other images
#  ====================================================================================================

import os
import pandas as pd
import shutil 

def clean_RDC(images_path, csv_path):
    print(f"\nCleaning: {images_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Keep only the columns we need
    df_clean = df[['ID', 'DR', 'ARMD']]

    # 2. Filter rows where DR or ARMD is 1
    df_filtered = df_clean[(df_clean['DR'] == 1) | (df_clean['ARMD'] == 1)]

    # 3. Build whitelist of allowed image filenames
    keep = set(df_filtered['ID'].astype(str) + ".png")

    print(f"Images to keep: {len(keep)}")

    # Save cleaned CSV (overwrite original)
    cleaned_csv_path = csv_path.replace(".csv", "_cleaned.csv")
    df_filtered.to_csv(cleaned_csv_path, index=False)
    print(f"Saved cleaned CSV → {cleaned_csv_path}")

    # Example: train/train/, test/test/, evaluation/evaluation/
    print(f"Looking inside: {images_path}")

    # 5. Delete images not in whitelist
    for filename in os.listdir(images_path):
        if filename.endswith(".png"):
            if filename not in keep:
                os.remove(os.path.join(images_path, filename))

    print("Done.")


def move_rename_RDC(images_source_path, images_destination_path, csv_path, dataset_name):
    df = pd.read_csv(csv_path)

    dr_list = set(df[(df["DR"]==1)]["ID"].astype(str) + ".png")
    amd_list = set(df[(df["ARMD"]==1)]["ID"].astype(str) + ".png")

    for image_name in dr_list:
        try:
            shutil.copy2(os.path.join(images_source_path, image_name), os.path.join(images_destination_path, "DR"))
            shutil.move(os.path.join(images_destination_path,"DR",image_name), os.path.join(images_destination_path,"DR", f"{dataset_name}_DR_{image_name}"))
        
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
    
    print("copy and rename success!")



#cleaning test data
clean_RDC("C:/Users/aryan/Projects/Major/raw/Retinal_Disease_Classification/Dataset/Test_Set/Test_Set/Test","C:/Users/aryan/Projects/Major/raw/Retinal_Disease_Classification/Dataset/Test_Set/Test_Set/RFMiD_Testing_Labels.csv")

#cleaning evaluation data
clean_RDC("C:/Users/aryan/Projects/Major/raw/Retinal_Disease_Classification/Dataset/Evaluation_Set/Evaluation_Set/Validation","C:/Users/aryan/Projects/Major/raw/Retinal_Disease_Classification/Dataset/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv")

#cleaning training data
clean_RDC("C:/Users/aryan/Projects/Major/raw/Retinal_Disease_Classification/Dataset/Training_Set/Training_Set/Training","C:/Users/aryan/Projects/Major/raw/Retinal_Disease_Classification/Dataset/Training_Set/Training_Set/RFMiD_Training_Labels.csv")


#moving evaluation set
move_rename_RDC(
    r"C:\Users\aryan\Projects\Major\raw\Retinal_Disease_Classification\Dataset\Evaluation_Set\Evaluation_Set\Validation",
    r"C:\Users\aryan\Projects\Major\data\images",
    r"C:\Users\aryan\Projects\Major\raw\Retinal_Disease_Classification\Dataset\Evaluation_Set\Evaluation_Set\RFMiD_Validation_Labels_cleaned.csv",
    "RDC")   
     
#moving test set
move_rename_RDC(
    r"C:\Users\aryan\Projects\Major\raw\Retinal_Disease_Classification\Dataset\Test_Set\Test_Set\Test",
    r"C:\Users\aryan\Projects\Major\data\images",
    r"C:\Users\aryan\Projects\Major\raw\Retinal_Disease_Classification\Dataset\Test_Set\Test_Set\RFMiD_Testing_Labels_cleaned.csv",
    "RDC")    
    
#moving training set
move_rename_RDC(
    r"C:\Users\aryan\Projects\Major\raw\Retinal_Disease_Classification\Dataset\Training_Set\Training_Set\Training",
    r"C:\Users\aryan\Projects\Major\data\images",
    r"C:\Users\aryan\Projects\Major\raw\Retinal_Disease_Classification\Dataset\Training_Set\Training_Set\RFMiD_Training_Labels_cleaned.csv",
    "RDC")        
