from move_images_and_rename import move_images_and_rename
import os
import shutil


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
clean_AMDnet23(r"C:\Users\aryan\Projects\Major\raw\AMDnet23\AMDNet23 Fundus Image Dataset for  Age-Related Macular Degeneration Disease Detection\AMDNet23 Dataset\train")

# for valid
clean_AMDnet23(r"C:\Users\aryan\Projects\Major\raw\AMDnet23\AMDNet23 Fundus Image Dataset for  Age-Related Macular Degeneration Disease Detection\AMDNet23 Dataset\valid")


# moving
splits = ["train", "valid"]

for split in splits:
    # for amd
    move_images_and_rename(
        os.path.join(r"C:\Users\aryan\Projects\Major\raw\AMDnet23\AMDNet23 Fundus Image Dataset for  Age-Related Macular Degeneration Disease Detection\AMDNet23 Dataset", split, "amd"),
        r"C:\Users\aryan\Projects\Major\data\images\AMD",
        "AMDnet23"
    )

    # for normal
    move_images_and_rename(
        os.path.join(r"C:\Users\aryan\Projects\Major\raw\AMDnet23\AMDNet23 Fundus Image Dataset for  Age-Related Macular Degeneration Disease Detection\AMDNet23 Dataset", split, "normal"),
        r"C:\Users\aryan\Projects\Major\data\images\Normal",
        "AMDnet23"
    )

print("AMDnet23 data transfer complete")
