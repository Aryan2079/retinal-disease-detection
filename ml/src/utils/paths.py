from pathlib import Path
from src.utils.config import load_cfg


cfg = load_cfg()

PROJECT_PATH = Path(__file__).resolve().parents[2]

DATA_DIR_PATH = PROJECT_PATH / "data"
SCRIPTS_DIR_PATH = PROJECT_PATH / "scripts"
SRC_DIR_PATH = PROJECT_PATH / "src"
CONFIG_PATH = Path(PROJECT_PATH / "config" / "config.yaml")

PROCESSED_DATA_PATH =  DATA_DIR_PATH / cfg["data"]["processed_dir"]
AMD_PROCESSED_PATH = PROCESSED_DATA_PATH / "AMD"
DR_PROCESSED_PATH = PROCESSED_DATA_PATH / "DR"
GLAUCOMA_PROCESSED_PATH = PROCESSED_DATA_PATH / "GLAUCOMA"
NORMAL_PROCESSED_PATH = PROCESSED_DATA_PATH / "NORMAL"

RAW_DATA_PATH = DATA_DIR_PATH / cfg["data"]["raw_dir"]
AMDnet23_RAW_DATA_PATH = RAW_DATA_PATH / "AMDnet23" / "AMDNet23 Fundus Image Dataset for  Age-Related Macular Degeneration Disease Detection" / "AMDNet23 Dataset"
EDC_RAW_DATA_PATH = RAW_DATA_PATH / "Eye_Disease_Classification" / "Dataset"
FDR_RAW_DATA_PATH = RAW_DATA_PATH / "Fundus_DR" / "Dataset" / "split_dataset"
GID_RAW_DATA_PATH = RAW_DATA_PATH / "Glaucoma_Fundus_Imaging_Dataset" / "Dataset"
MDG_RAW_DATA_PATH = RAW_DATA_PATH / "Macular_Disease_Detection" / "Macular Degeneration Disease Dataset"
OD_RAW_DATA_PATH = RAW_DATA_PATH / "Ocular_Dataset" / "preprocessed"
ODD_RAW_DATA_PATH = RAW_DATA_PATH / "Ocular_Disease_Detection"
RDC_RAW_DATA_PATH = RAW_DATA_PATH / "Retinal_Disease_Classification" / "Dataset"
RFI50_RAW_DATA_PATH = RAW_DATA_PATH / "Retinal_Fundus_Image_50k" / "Retinal Fundus Images"
SGD_RAW_DATA_PATH = RAW_DATA_PATH / "Standarized_Glaucoma_Dataset" / "Dataset"