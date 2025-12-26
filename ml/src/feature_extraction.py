import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm
import random
import os

# ============================================================
# Paths (same as training script)
# ============================================================
TRAIN_DIR = r"C:\Users\aryan\Projects\Major\preproc\train"
VAL_DIR   = r"C:\Users\aryan\Projects\Major\preproc\val"
TEST_DIR  = r"C:\Users\aryan\Projects\Major\preproc\test"

MODEL_PATH = r"C:\Users\aryan\Projects\Major\models\efficientnet_b3_retinal_disease_cnn_v1.pth"

SAVE_DIR = "features"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# Transforms (EXACTLY same as training)
# ============================================================
val_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# Reconstruct MobileNetV2 exactly like in training
# ============================================================
weights = MobileNet_V2_Weights.DEFAULT
cnn = mobilenet_v2(weights=weights)

# Replace classifier head (same as training)
in_features = cnn.classifier[1].in_features
cnn.classifier[1] = nn.Linear(in_features, 4)

# Load trained weights
cnn.load_state_dict(torch.load(MODEL_PATH, map_location=device))
cnn.to(device)

print("Model loaded.")

# Freeze everything
for p in cnn.parameters():
    p.requires_grad = False

cnn.eval()

# Extractor = feature layers only
feature_extractor = cnn.features.to(device)
feature_extractor.eval()

# ============================================================
# Load datasets
# ============================================================
train_dataset_full = datasets.ImageFolder(TRAIN_DIR, transform=val_transforms)
val_dataset_full   = datasets.ImageFolder(VAL_DIR,   transform=val_transforms)
test_dataset_full  = datasets.ImageFolder(TEST_DIR,  transform=val_transforms)

# ============================================================
# Rebuild the SAME subset used in training (10%)
# ============================================================
fraction = 0.1
subset_size = int(len(train_dataset_full) * fraction)

train_indices = list(range(len(train_dataset_full)))
random.shuffle(train_indices)
train_indices = train_indices[:subset_size]

train_subset = Subset(train_dataset_full, train_indices)

# ============================================================
# Rebuild reduced VAL subset
# ============================================================
val_subset_size = max(2000, int(len(val_dataset_full) * 0.10))
val_indices = list(range(len(val_dataset_full)))
random.shuffle(val_indices)
val_indices = val_indices[:val_subset_size]

val_subset = Subset(val_dataset_full, val_indices)

# ============================================================
# Dataloaders
# ============================================================
train_loader = DataLoader(train_subset, batch_size=32, shuffle=False)
val_loader   = DataLoader(val_subset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset_full, batch_size=32, shuffle=False)


# ============================================================
# Feature extraction function
# ============================================================
def extract(loader, name):
    all_feats = []
    all_labels = []

    print(f"\nExtracting {name} features...")

    with torch.no_grad():
        prog = tqdm(loader, unit="batch")

        for images, labels in prog:
            images = images.to(device)

            feats = feature_extractor(images)        # [B, 1280, 16, 16] for 512 input
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            feats = feats.view(feats.size(0), -1)    # [B, 1280]

            all_feats.append(feats.cpu())
            all_labels.append(labels)

    features = torch.cat(all_feats)
    labels   = torch.cat(all_labels)

    torch.save({"features": features, "labels": labels}, f"{SAVE_DIR}/{name}_features.pt")
    print(f"{name} saved → {SAVE_DIR}/{name}_features.pt")


# ============================================================
# Run extraction
# ============================================================
extract(train_loader, "train")
extract(val_loader, "val")
extract(test_loader, "test")

print("\n🔥 All feature sets extracted successfully.")
