import os
import json
import imagehash
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import defaultdict

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
BASE_DIR = r"C:\Users\aryan\Projects\Major\data\images"
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
HASH_SIZE = 16
# -----------------------------------------------

def hash_image(path):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            phash = imagehash.phash(img, hash_size=HASH_SIZE)
            return (path, str(phash))
    except Exception:
        return (path, None)

def get_all_images(dataset_dir):
    image_paths = []

    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(VALID_EXT):
                full_path = os.path.join(root, f)
                image_paths.append(full_path)

    return image_paths

def main():
    print("\n📌 Scanning images recursively...")
    paths = get_all_images(BASE_DIR)
    total_images = len(paths)
    print(f"✅ Total images found: {total_images}")
    print(f"Using {cpu_count()} CPU cores for hashing...\n")

    hashes = {}
    with Pool(cpu_count()) as p:
        for path, ph in tqdm(
            p.imap_unordered(hash_image, paths),
            total=total_images,
            desc="Hashing images",
            ncols=90
        ):
            hashes[path] = ph

    groups = defaultdict(list)
    for path, ph in hashes.items():
        if ph is not None:
            groups[ph].append(path)

    duplicate_groups = [imgs for imgs in groups.values() if len(imgs) > 1]
    total_duplicate_images = sum(len(g) for g in duplicate_groups)

    # ---------------------------------------------------
    # SAVE TO JSON
    # ---------------------------------------------------
    json_data = {
        "base_dir": BASE_DIR,
        "total_images": total_images,
        "unique_hashes": len(groups),
        "duplicate_groups_count": len(duplicate_groups),
        "total_duplicate_images": total_duplicate_images,
        "duplicate_groups": duplicate_groups
    }

    with open("duplicate_report.json", "w") as f:
        json.dump(json_data, f, indent=2)

    print("\n----------------------------------------")
    print("         🔍 DUPLICATE REPORT")
    print("----------------------------------------")
    print(f"Total images scanned: {total_images}")
    print(f"Unique images (unique hashes): {len(groups)}")
    print(f"Duplicate groups found: {len(duplicate_groups)}")
    print(f"Total images involved in duplicate groups: {total_duplicate_images}")
    print("Saved full duplicate details to: duplicate_report.json")
    print("SAFE: No files were deleted.")
    print("----------------------------------------\n")

if __name__ == "__main__":
    main()
