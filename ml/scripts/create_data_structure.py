from pathlib import Path

def create_data_structure(root_dir="data"):
    root = Path(root_dir)

    folders = [
        root / "raw_data" / "AMDnet23",
        root / "raw_data" / "Eye_Disease_Classification",
        root / "raw_data" / "EyePacs_DR",
        root / "raw_data" / "Fundus_DR",
        root / "raw_data" / "Glaucoma_Fundus_Imaging_Dataset",
        root / "raw_data" / "Macular_Disease_Detection",
        root / "raw_data" / "Ocular_Dataset",
        root / "raw_data" / "Ocular_Disease_Detection",
        root / "raw_data" / "Retinal_Disease_Classification",
        root / "raw_data" / "Retinal_Fundus_Image",
        root / "raw_data" / "Retinal_Fundus_Image_50k",
        root / "raw_data" / "Standarized_Glaucoma_Dataset",
        root / "processed_data",
        root / "splits",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    print("✅ Data directory structure created successfully.")

if __name__ == "__main__":
    create_data_structure()
