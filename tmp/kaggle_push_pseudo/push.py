#!/usr/bin/env python3
"""Push pseudo-label generator notebook to Kaggle."""
import json
import sys
from pathlib import Path
import requests

HERE = Path(__file__).parent
meta = json.loads((HERE / "kernel-metadata.json").read_text())
nb_path = HERE / meta["code_file"]
nb_json = json.loads(nb_path.read_text())

if "cells" in nb_json:
    for cell in nb_json["cells"]:
        if "outputs" in cell and cell.get("cell_type") == "code":
            cell["outputs"] = []
        if "source" in cell and isinstance(cell["source"], list):
            cell["source"] = "".join(cell["source"])

creds = json.loads(Path.home().joinpath(".kaggle/kaggle.json").read_text())
KEY = creds["key"].strip()

payload = {
    "slug": meta["id"],
    "newTitle": meta["title"],
    "text": json.dumps(nb_json),
    "language": meta["language"],
    "kernelType": meta["kernel_type"],
    "isPrivate": meta["is_private"],
    "enableGpu": meta["enable_gpu"],
    "enableTpu": meta["enable_tpu"],
    "enableInternet": meta["enable_internet"],
    "datasetDataSources": meta["dataset_sources"],
    "competitionDataSources": meta["competition_sources"],
    "kernelDataSources": meta["kernel_sources"],
    "modelDataSources": meta["model_sources"],
    "categoryIds": [],
}

resp = requests.post(
    "https://www.kaggle.com/api/v1/kernels/push",
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    json=payload, timeout=300,
)
print(f"Status: {resp.status_code}")
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
body = resp.json() if resp.status_code == 200 else {}
sys.exit(0 if resp.status_code == 200 and not body.get("hasError") else 1)
