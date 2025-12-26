import shutil
import os

def move_images_and_rename(source_image_path, destination_image_path, dataset_name):
    image_list = os.listdir(source_image_path)

    for image in image_list:
        try:
            shutil.copy2(os.path.join(source_image_path, image),destination_image_path)

            shutil.move(os.path.join(destination_image_path, image), os.path.join(destination_image_path, f"{dataset_name}_{os.path.basename(destination_image_path)}_{image}") )

        except FileExistsError:
            print("file doesnt exist")
            return
        except Exception as e:
            print(f"an exception occured {e}")
            return

    print("File copy success!")