import yaml
from paths import DATA_DIR_PATH, CONFIG_PATH

def load_cfg():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)