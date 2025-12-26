from move_images_and_rename import move_images_and_rename
import os


splits = ["train", "val"]

for split in splits:
    # for amd
    move_images_and_rename(
        os.path.join(r"C:\Users\aryan\Projects\Major\raw\Macular_Disease_Detection\Macular Degeneration Disease Dataset", split, "amd"),
        r"C:\Users\aryan\Projects\Major\data\images\AMD",
        "MDG"
    )

    # for normal
    move_images_and_rename(
        os.path.join(r"C:\Users\aryan\Projects\Major\raw\Macular_Disease_Detection\Macular Degeneration Disease Dataset", split, "normal"),
        r"C:\Users\aryan\Projects\Major\data\images\Normal",
        "MDG"
    )

print("MDG data transfer complete")
