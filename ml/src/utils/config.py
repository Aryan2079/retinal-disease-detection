import yaml
from pathlib import Path

def load_cfg():
    config_path = Path("../../config/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)