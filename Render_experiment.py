import json
from jinja2 import Template
from pathlib import Path

DATA_DIR = Path("experiments/data")
TEMPLATE_PATH = Path("experiments/templates/experiment.html")
OUTPUT_DIR = Path("experiments/output")

# ✅ create directories safely
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

template = Template(TEMPLATE_PATH.read_text())

for json_file in DATA_DIR.glob("*.json"):
    data = json.loads(json_file.read_text())

    html = template.render(**data)

    output_file = OUTPUT_DIR / f"{data['experiment_id']}.html"
    output_file.write_text(html)

    print(f"Rendered: {output_file}")
