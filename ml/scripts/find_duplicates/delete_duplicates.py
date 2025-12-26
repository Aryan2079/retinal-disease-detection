import os
import json

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
JSON_PATH = r"C:\Users\aryan\Projects\Major\duplicate_report.json"    # the file created earlier
LOG_PATH = r"C:\Users\aryan\Projects\Major\deleted_duplicates.txt"    # log of deleted images
# ----------------------------------------------------


# Load duplicate detection data
with open(JSON_PATH, "r") as f:
    data = json.load(f)

BASE_DIR = data["base_dir"]
duplicate_groups = data["duplicate_groups"]

deleted = 0
missing = 0

# Create log file
log_file = open(LOG_PATH, "w", encoding="utf-8")
log_file.write("Deleted duplicate files:\n\n")

print("\n-----------------------------------------")
print("     🗑️  STARTING DUPLICATE DELETION")
print("-----------------------------------------")

for group in duplicate_groups:
    # Keep the first image in each group
    keep = group[0]
    to_delete = group[1:]

    for img_path in to_delete:
        abs_path = img_path  # JSON already stores absolute paths

        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
                deleted += 1
                log_file.write(abs_path + "\n")
            except Exception as e:
                print(f"[ERROR] Could not delete {abs_path}: {e}")
        else:
            missing += 1

log_file.close()

print("\n-----------------------------------------")
print("         🧹 DUPLICATE CLEAN REPORT")
print("-----------------------------------------")
print(f"Duplicate groups processed: {len(duplicate_groups)}")
print(f"Images deleted: {deleted}")
print(f"Missing (already removed or moved earlier): {missing}")
print(f"Deletion log saved to: {LOG_PATH}")
print("-----------------------------------------")
print("Done.\n")
