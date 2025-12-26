import os
import shutil
import random
from math import floor

# -----------------------------
# CONFIG
# -----------------------------
SOURCE_DIR = r"C:\Users\aryan\Projects\Major\data\images"   # <-- your folder with AMD/DR/etc
TARGET_DIR = r"C:\Users\aryan\Projects\Major\preproc"    # <-- output here

CLASSES = ["AMD", "DR", "Glaucoma", "Normal"]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# -----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def stratified_split():
    print("\n🔍 Starting stratified split...\n")

    # Create target folders
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            ensure_dir(os.path.join(TARGET_DIR, split, cls))

    summary = {}

    for cls in CLASSES:
        cls_path = os.path.join(SOURCE_DIR, cls)
        images = [f for f in os.listdir(cls_path)
                  if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))]

        random.shuffle(images)

        total = len(images)
        n_train = floor(total * TRAIN_RATIO)
        n_val   = floor(total * VAL_RATIO)
        n_test  = total - n_train - n_val

        summary[cls] = {
            "total": total,
            "train": n_train,
            "val": n_val,
            "test": n_test
        }

        # Split
        train_imgs = images[:n_train]
        val_imgs   = images[n_train:n_train+n_val]
        test_imgs  = images[n_train+n_val:]

        # Copy to new folders
        for fname in train_imgs:
            shutil.copy(
                os.path.join(cls_path, fname),
                os.path.join(TARGET_DIR, "train", cls, fname)
            )

        for fname in val_imgs:
            shutil.copy(
                os.path.join(cls_path, fname),
                os.path.join(TARGET_DIR, "val", cls, fname)
            )

        for fname in test_imgs:
            shutil.copy(
                os.path.join(cls_path, fname),
                os.path.join(TARGET_DIR, "test", cls, fname)
            )

    # -----------------------------
    # PRINT SUMMARY
    # -----------------------------
    print("\n-----------------------------")
    print("     📊 SPLIT SUMMARY")
    print("-----------------------------")

    grand_train = grand_val = grand_test = 0

    for cls, stats in summary.items():
        print(f"\nClass: {cls}")
        print(f" Total: {stats['total']}")
        print(f" Train: {stats['train']}")
        print(f" Val:   {stats['val']}")
        print(f" Test:  {stats['test']}")

        grand_train += stats['train']
        grand_val   += stats['val']
        grand_test  += stats['test']

    print("\n-----------------------------")
    print("     📦 TOTAL IMAGES")
    print("-----------------------------")
    print(f"Train: {grand_train}")
    print(f"Val:   {grand_val}")
    print(f"Test:  {grand_test}")
    print("\n✅ Done!")

# -----------------------------
if __name__ == "__main__":
    stratified_split()
