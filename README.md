# Deep Learning for Glaucoma Detection using Retinal Fundus Images

This project investigates multiple deep learning architectures for automated glaucoma analysis using retinal fundus images. The work includes semantic segmentation with UNet variants and image classification using Vision Transformers.

---

## Overview

The goal of this project was to explore different computer vision approaches for glaucoma detection and compare traditional convolutional segmentation models with transformer-based image classification.

The project includes:

- Medical image preprocessing
- Semantic segmentation
- Attention-based segmentation
- Vision Transformer classification

---

## Project Structure

```
.
├── notebooks/
│   ├── unet_segmentation.ipynb
│   ├── attention_unet.ipynb
│   └── vision_transformer.ipynb
│
├── README.md
└── requirements.txt
```

---

## Implemented Work

### 1. UNet Segmentation

- Built a baseline UNet model
- Image preprocessing
- Data augmentation
- Semantic segmentation training

---

### 2. Attention UNet

- Implemented Attention Gates
- Improved feature selection
- Glaucoma optic disc/cup segmentation
- Model training and validation

---

### 3. Vision Transformer

- Fine-tuned a pretrained Vision Transformer
- Binary glaucoma classification
- Performance evaluation using multiple classification metrics

---

## Models

- UNet
- Attention UNet
- Vision Transformer (ViT)

---

## Dataset

Retinal Fundus Images for Glaucoma Detection

---

## Technologies

- Python
- PyTorch
- OpenCV
- Albumentations
- timm
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## Workflow

```
Fundus Images
      │
      ▼
Image Preprocessing
      │
      ▼
Data Augmentation
      │
      ├──────────────┐
      ▼              ▼
UNet         Vision Transformer
      │              │
Attention UNet       Classification
      │              │
      └──────┬───────┘
             ▼
      Model Evaluation
```

---

## Evaluation

Segmentation:

- Dice Score
- IoU
- Visual mask comparison

Classification:

- Accuracy
- ROC-AUC
- Confusion Matrix
- Classification Report

---

## Skills Demonstrated

- Medical Image Analysis
- Semantic Segmentation
- Vision Transformers
- Transfer Learning
- Image Augmentation
- Deep Learning with PyTorch
- Computer Vision
- Model Evaluation

---

## Future Improvements

- Cross-validation
- Hyperparameter optimization
- Ensemble learning
- Model explainability using Grad-CAM
- Deployment as a web application

---

## Author

Personal deep learning project exploring segmentation and transformer-based approaches for glaucoma detection from retinal fundus images.
