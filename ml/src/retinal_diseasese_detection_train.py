import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
import os
import random
from tqdm import tqdm


def main():
    # ============================================================
    # Configurations
    # ============================================================
    BATCH_SIZE = 16
    NUM_CLASSES = 4
    NUM_EPOCHS = 10
    MODEL_SAVE_PATH = r"C:\Users\aryan\Projects\Major\models\efficientnet_b3_retinal_disease_cnn_v1.pth"

    # Directories for your splits
    train_dir = r"C:\Users\aryan\Projects\Major\preproc\train"
    val_dir   = r"C:\Users\aryan\Projects\Major\preproc\val"
    test_dir  = r"C:\Users\aryan\Projects\Major\preproc\test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ============================================================
    # Transforms
    # ============================================================
    train_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    ])

    val_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ============================================================
    # Datasets
    # ============================================================
    train_dataset_full = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset_full   = datasets.ImageFolder(val_dir, transform=val_transforms)
    test_dataset_full  = datasets.ImageFolder(test_dir, transform=val_transforms)


    # ============================================================
    # Weighted Sampling for Balanced Training
    # ============================================================
    targets = [label for _, label in train_dataset_full.samples]
    class_sample_count = np.bincount(targets)
    class_weights = 1.0 / class_sample_count

    samples_weight = np.array([class_weights[t] for t in targets])
    samples_weight = torch.from_numpy(samples_weight).float()

    # ============================================================
    # create subset of the dataset
    # ============================================================
    fraction = 0.1
    subset_size = int(len(train_dataset_full) * fraction)
    indices = list(range(len(train_dataset_full)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]
    train_dataset_subset = Subset(train_dataset_full, subset_indices)

    # ----------------------------
    # STEP 3: filter weights for subset only
    # ----------------------------
    subset_weights = samples_weight[subset_indices]

    sampler = WeightedRandomSampler(weights=subset_weights,
                                    num_samples=len(subset_weights),
                                    replacement=True)

    train_loader = DataLoader(train_dataset_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)

    # Make VAL smaller too for CPU
    val_subset_size = max(2000, int(len(val_dataset_full) * 0.10))  # max 2k samples or 10%
    val_indices = list(range(len(val_dataset_full)))
    random.shuffle(val_indices)
    val_indices = val_indices[:val_subset_size]
    val_dataset = Subset(val_dataset_full, val_indices)

    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ============================================================
    # Model (EfficientNet-B3)
    # ============================================================
    weights = MobileNet_V2_Weights.DEFAULT
    cnn = models.mobilenet_v2(weights=weights)

    # Replace classifier head (for 4-class classification)
    in_features = cnn.classifier[1].in_features
    cnn.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    cnn.to(device)

    # Freeze all convolution blocks
    for param in cnn.features.parameters():
        param.requires_grad = False

    # Unfreeze the last convolution block for slight fine-tuning
    for param in cnn.features[-1].parameters():
        param.requires_grad = True

    # ============================================================
    # Loss & Optimizer
    # ============================================================
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, cnn.parameters()),
                            lr=1e-4, weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # ============================================================
    # Training Loop With tqdm Progress Bar
    # ============================================================
    for epoch in range(NUM_EPOCHS):
        cnn.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")

        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} → Avg Training Loss: {avg_loss:.4f}")

        scheduler.step()

        # ========================================================
        # Validation Step
        # ========================================================
        cnn.eval()
        correct = 0
        total = 0
        val_loss = 0

        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", unit="batch")

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = cnn(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                val_progress.set_postfix({
                "loss": f"{val_loss/len(val_progress):.4f}",
                "acc": f"{correct/total*100:.2f}%"
                })

        val_loss /= len(val_loader)
        accuracy = correct / total * 100

        print(f"Validation Loss: {val_loss:.4f} | Accuracy: {accuracy:.2f}%\n")

    # ============================================================
    # Save Model
    # ============================================================
    torch.save(cnn.state_dict(), MODEL_SAVE_PATH)
    print("Model saved.")

if __name__ == "__main__":  
    main()