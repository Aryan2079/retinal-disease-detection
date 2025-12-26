import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import DeiTForImageClassification, DeiTConfig

# ============================================================
# Dataset for CNN-extracted feature tensors
# ============================================================
class FeatureDataset(Dataset):
    def __init__(self, features_path):
        # Load the pre-saved tensors with weights_only=False for backward compatibility
        data = torch.load(features_path, weights_only=False)
        
        # Handle different possible formats
        if isinstance(data, dict):
            # If saved as dictionary
            self.features = data['features']
            self.labels = data['labels']
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            # If saved as tuple/list
            self.features, self.labels = data
        else:
            raise ValueError(f"Unexpected data format in {features_path}. "
                           f"Expected dict or tuple, got {type(data)}")
        
        # Ensure correct types
        self.features = self.features.float()
        if self.labels.dtype != torch.long:
            self.labels = self.labels.long()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================
# Option 1: Reshape 1D features to 2D (if you know the spatial dims)
# ============================================================
class HybridDeiTWithReshape(nn.Module):
    def __init__(self, cnn_feat_dim=1280, spatial_size=8, num_classes=4):
        """
        Args:
            cnn_feat_dim: dimension of flattened CNN features (e.g., 1280)
            spatial_size: sqrt of number of spatial locations (e.g., 8 for 8x8 = 64)
            num_classes: number of output classes
        """
        super().__init__()
        
        self.spatial_size = spatial_size
        # Calculate channels: 1280 = C * 8 * 8, so C = 1280 / 64 = 20
        self.channels = cnn_feat_dim // (spatial_size * spatial_size)
        
        if self.channels * spatial_size * spatial_size != cnn_feat_dim:
            raise ValueError(f"Cannot reshape {cnn_feat_dim} into {spatial_size}x{spatial_size} grid")

        # Create DeiT-Tiny backbone
        config = DeiTConfig.from_pretrained("facebook/deit-tiny-patch16-224")
        config.num_labels = num_classes
        self.deit = DeiTForImageClassification(config)

        # Project CNN feature maps → DeiT embedding dim (192)
        self.proj = nn.Conv2d(self.channels, config.hidden_size, kernel_size=1)

    def forward(self, feats):
        # feats shape: [B, 1280]
        B = feats.shape[0]
        
        # Reshape to [B, C, H, W]
        feats = feats.view(B, self.channels, self.spatial_size, self.spatial_size)
        
        x = self.proj(feats)  # [B, 192, H, W]

        # Flatten spatial map → patch embeddings
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # → [B, HW, 192]

        outputs = self.deit(
            pixel_values=None,
            encoder_hidden_states=x,
            return_dict=True
        )
        return outputs.logits


# ============================================================
# Option 2: Simple MLP + Transformer (recommended for 1D features)
# ============================================================
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, feat_dim=1280, num_classes=4, hidden_dim=192, num_heads=3, num_layers=4):
        super().__init__()
        
        # Project features to transformer dimension
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, feats):
        # feats shape: [B, feat_dim]
        B = feats.shape[0]
        
        # Project to hidden dim and add batch dimension for sequence
        x = self.input_proj(feats).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 2, hidden_dim]
        
        # Apply transformer
        x = self.transformer(x)  # [B, 2, hidden_dim]
        
        # Use CLS token for classification
        cls_output = x[:, 0]  # [B, hidden_dim]
        
        return self.classifier(cls_output)


# ============================================================
# Training Script
# ============================================================
def train_deit_classifier():
    # Paths to feature files
    train_feats_path = r"C:\Users\aryan\Projects\Major\features\train_features.pt"
    val_feats_path   = r"C:\Users\aryan\Projects\Major\features\val_features.pt"
    save_path        = r"C:\Users\aryan\Projects\Major\models\deit_classifier_v1.pth"

    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 3e-4
    
    # Choose which model to use:
    USE_SIMPLE_TRANSFORMER = True  # Set to False to try reshape approach

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create datasets and dataloaders
    try:
        train_set = FeatureDataset(train_feats_path)
        val_set   = FeatureDataset(val_feats_path)
        
        print(f"Train set size: {len(train_set)}")
        print(f"Val set size: {len(val_set)}")
        print(f"Feature shape: {train_set[0][0].shape}")
        print(f"Label shape: {train_set[0][1].shape if hasattr(train_set[0][1], 'shape') else 'scalar'}")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("\nTrying to inspect the file contents...")
        data = torch.load(train_feats_path, weights_only=False)
        print(f"Data type: {type(data)}")
        if isinstance(data, dict):
            print(f"Dictionary keys: {data.keys()}")
        elif isinstance(data, (list, tuple)):
            print(f"List/Tuple length: {len(data)}")
            for i, item in enumerate(data):
                print(f"  Item {i} type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
        raise

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model based on feature dimensions
    feat_dim = train_set[0][0].shape[0]
    
    if USE_SIMPLE_TRANSFORMER:
        print(f"\nUsing SimpleTransformerClassifier with feat_dim={feat_dim}")
        model = SimpleTransformerClassifier(feat_dim=feat_dim, num_classes=4).to(device)
    else:
        # Try to automatically determine spatial size
        # Common sizes: 7x7=49, 8x8=64, 10x10=100, 14x14=196
        import math
        possible_sizes = [7, 8, 10, 14, 16]
        spatial_size = None
        for size in possible_sizes:
            if feat_dim % (size * size) == 0:
                spatial_size = size
                break
        
        if spatial_size is None:
            raise ValueError(f"Cannot determine spatial size for feat_dim={feat_dim}. Use SimpleTransformerClassifier instead.")
        
        print(f"\nUsing HybridDeiTWithReshape with spatial_size={spatial_size}")
        model = HybridDeiTWithReshape(cnn_feat_dim=feat_dim, spatial_size=spatial_size, num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for feats, labels in train_bar:
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} → Avg Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")

        with torch.no_grad():
            for feats, labels in val_bar:
                feats, labels = feats.to(device), labels.to(device)

                outputs = model(feats)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                val_bar.set_postfix({
                    "loss": f"{val_loss/(val_bar.n+1):.4f}",
                    "acc": f"{(correct/total)*100:.2f}%"
                })

        val_acc = (correct/total)*100
        print(f"Validation Loss: {val_loss/len(val_loader):.4f} | Accuracy: {val_acc:.2f}%\n")

    # Save model
    torch.save(model.state_dict(), save_path)
    print("Model saved to:", save_path)


if __name__ == "__main__":
    train_deit_classifier()