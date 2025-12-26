import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Dataset (same as training)
# ============================================================
class FeatureDataset(Dataset):
    def __init__(self, features_path):
        data = torch.load(features_path, weights_only=False)
        
        if isinstance(data, dict):
            self.features = data['features']
            self.labels = data['labels']
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            self.features, self.labels = data
        else:
            raise ValueError(f"Unexpected data format. Expected dict or tuple, got {type(data)}")
        
        self.features = self.features.float()
        if self.labels.dtype != torch.long:
            self.labels = self.labels.long()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================
# Model Architecture (must match training)
# ============================================================
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, feat_dim=1280, num_classes=4, hidden_dim=192, num_heads=3, num_layers=4):
        super().__init__()
        
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, feats):
        B = feats.shape[0]
        x = self.input_proj(feats).unsqueeze(1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        cls_output = x[:, 0]
        return self.classifier(cls_output)


# ============================================================
# Evaluation Functions
# ============================================================
def evaluate_model(model, dataloader, device, class_names=None):
    """
    Evaluate model and return predictions, labels, and metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for feats, labels in tqdm(dataloader, desc="Evaluating"):
            feats, labels = feats.to(device), labels.to(device)
            
            outputs = model(feats)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\n")
    
    # Classification report
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(all_labels)))]
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return all_preds, all_labels, all_probs


def plot_confusion_matrix(labels, predictions, class_names=None, save_path=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(labels, predictions)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_per_class_metrics(labels, predictions, class_names=None, save_path=None):
    """
    Plot per-class precision, recall, and F1-score
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(precision))]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to: {save_path}")
    
    plt.show()


def analyze_misclassifications(labels, predictions, probs, class_names=None, top_k=10):
    """
    Analyze the most confident misclassifications
    """
    misclassified_idx = np.where(labels != predictions)[0]
    
    if len(misclassified_idx) == 0:
        print("No misclassifications found!")
        return
    
    # Get confidence for misclassified samples
    misclassified_confs = np.max(probs[misclassified_idx], axis=1)
    
    # Sort by confidence (most confident mistakes first)
    sorted_idx = np.argsort(misclassified_confs)[::-1][:top_k]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(probs.shape[1])]
    
    print("\n" + "="*60)
    print(f"TOP {top_k} MOST CONFIDENT MISCLASSIFICATIONS")
    print("="*60)
    
    for rank, idx in enumerate(sorted_idx, 1):
        sample_idx = misclassified_idx[idx]
        true_label = labels[sample_idx]
        pred_label = predictions[sample_idx]
        confidence = misclassified_confs[idx]
        
        print(f"\n{rank}. Sample #{sample_idx}")
        print(f"   True: {class_names[true_label]}")
        print(f"   Predicted: {class_names[pred_label]} (confidence: {confidence:.2%})")
        print(f"   All probabilities: {dict(zip(class_names, probs[sample_idx]))}")


# ============================================================
# Main Evaluation Script
# ============================================================
def main():
    # Configuration
    MODEL_PATH = r"C:\Users\aryan\Projects\Major\models\deit_classifier_v1.pth"
    VAL_FEATURES_PATH = r"C:\Users\aryan\Projects\Major\features\val_features.pt"
    TEST_FEATURES_PATH = r"C:\Users\aryan\Projects\Major\features\test_features.pt"  # if available
    
    # Output paths for plots
    RESULTS_DIR = r"C:\Users\aryan\Projects\Major\results"
    CM_SAVE_PATH = r"C:\Users\aryan\Projects\Major\results\confusion_matrix.png"
    METRICS_SAVE_PATH = r"C:\Users\aryan\Projects\Major\results\per_class_metrics.png"
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Class names (update these to match your dataset)
    CLASS_NAMES = ['Glaucoma', 'Normal', 'AMD', 'DR']
    
    BATCH_SIZE = 32
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load dataset
    print("Loading validation dataset...")
    val_dataset = FeatureDataset(VAL_FEATURES_PATH)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Feature dimension: {val_dataset[0][0].shape[0]}\n")
    
    # Initialize model
    feat_dim = val_dataset[0][0].shape[0]
    model = SimpleTransformerClassifier(feat_dim=feat_dim, num_classes=len(CLASS_NAMES))
    
    # Load trained weights
    print(f"Loading model from: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    print("Model loaded successfully!\n")
    
    # Evaluate
    predictions, labels, probs = evaluate_model(model, val_loader, device, CLASS_NAMES)
    
    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions, CLASS_NAMES, CM_SAVE_PATH)
    
    # Plot per-class metrics
    plot_per_class_metrics(labels, predictions, CLASS_NAMES, METRICS_SAVE_PATH)
    
    # Analyze misclassifications
    analyze_misclassifications(labels, predictions, probs, CLASS_NAMES, top_k=10)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()