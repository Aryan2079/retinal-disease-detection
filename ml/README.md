# 🧠 Retinal Disease Detection

This project focuses on detecting retinal diseases from fundus images using deep learning. It is designed to be **reproducible**, **cross-platform**, and **easy to set up** on Windows, Linux, and macOS.

> ⚠️ **IMPORTANT:** Datasets and trained models are NOT included in this repository.

---

## 📁 Project Structure

```
retinal-disease-detection/
├── data/                    # datasets (NOT included)
│   └── README.md
├── models/                  # trained models (auto-created)
│   └── README.md
├── results/                 # results / logs (auto-created)
│   └── README.md
├── src/                     # source code
│   └── utils/
├── scripts/                 # runnable scripts
├── configs/                 # configuration files
│   └── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Prerequisites

- Python 3.10
- Conda (recommended)
- Git

---

## 🚀 Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Aryan2079/retinal-disease-detection.git
cd retinal-disease-detection
```

---

### 2️⃣ Create and activate conda environment

```bash
conda create -n retinal python=3.10 -y
conda activate retinal
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Setup (MANDATORY)

> ⚠️ **Datasets are NOT included in this repository.**

After cloning, the `data/` folder will already exist, but it will be empty.

### Required folder structure:

```
data/
├── raw_data/
│   ├── AMDnet23/
│   ├── Eye_Disease_Classification/
│   ├── EyePacs_DR/
│   ├── Fundus_DR/
│   ├── Glaucoma_Fundus_Imaging_Dataset/
│   ├── Macular_Disease_Detection/
│   ├── Ocular_Dataset/
│   ├── Ocular_Disease_Detection/
│   ├── Retinal_Disease_Classification/
│   ├── Retinal_Fundus_Image/
│   ├── Retinal_Fundus_Image_50k/
│   └── Standarized_Glaucoma_Dataset/
├── processed_data/
└── splits/
```

### Steps:

1. **Create the data structure:**
   ```bash
   python scripts/create_data_structure.py
   ```

2. **Download datasets** and place them directly inside their corresponding folders in `data/raw_data/`

3. **Do NOT rename dataset folders**

4. **Do NOT push datasets to GitHub**

---

## 📥 Dataset Download Links

All dataset download sources are listed in:

```
dataset_links.yaml
```

Place each dataset inside the folder matching its name.

---

## ⚙️ Configuration

All project settings are defined in:

```
configs/config.yaml
```

This file controls:
- Data structure
- Image size
- Preprocessing flags
- Training parameters
- Model settings

> ❗ **Do NOT hardcode paths anywhere in the code.**

---

## ▶️ Running the Project

For now, only setup and dataset placement are required.

Preprocessing, training, and inference scripts will be added later and **MUST be run in order**.

---

## ❗ Important Rules (Read Carefully)

### ❌ Don't:
- Push datasets or trained models to GitHub
- Modify code inside `src/` unless explicitly assigned
- Run scripts out of order

### ✅ Do:
- Use GitHub branches and Pull Requests
- Follow this README exactly for reproducibility

---

## 📬 Contact

If something breaks after following the instructions exactly, raise a GitHub issue or contact the repository owner:

**Aryan Bhattarai**

---

## 📄 License

*Add your license information here*

---

## 🙏 Acknowledgments

*Add acknowledgments for datasets and resources used*