#!/usr/bin/env python3
"""Push EoS5 notebook v7 to Kaggle using Bearer auth (KGAT_ token compatible).

Key learnings:
- KGAT_ token requires Bearer auth (not Basic)
- slug field expects full "user/slug" format (matches meta_data['id'])
- Notebook cells: outputs must be cleared in code cells, source list joined to string
"""
import json
import sys
from pathlib import Path
import requests

HERE = Path(__file__).parent
META_PATH = HERE / "kernel-metadata.json"
NB_PATH = HERE / "birdclef-2026-eos-5.ipynb"

creds = json.loads(Path.home().joinpath(".kaggle/kaggle.json").read_text())
USERNAME = creds["username"]
KEY = creds["key"].strip()
assert KEY.startswith("KGAT_"), "Expected KGAT_ token"

meta = json.loads(META_PATH.read_text())
nb_json = json.loads(NB_PATH.read_text())

# Normalize notebook cells: clear code-cell outputs, join source lists into strings
if "cells" in nb_json:
    for cell in nb_json["cells"]:
        if "outputs" in cell and cell.get("cell_type") == "code":
            cell["outputs"] = []
        if "source" in cell and isinstance(cell["source"], list):
            cell["source"] = "".join(cell["source"])

nb_text = json.dumps(nb_json)

payload = {
    "slug": meta["id"],  # full "user/slug" form
    "newTitle": meta["title"],
    "text": nb_text,
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

url = "https://www.kaggle.com/api/v1/kernels/push"
headers = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

print(f"Pushing {meta['id']} (notebook size: {len(nb_text)} chars)...")
resp = requests.post(url, headers=headers, json=payload, timeout=300)
print(f"Status: {resp.status_code}")
try:
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
except Exception:
    print(resp.text)

body = resp.json() if resp.status_code == 200 else {}
sys.exit(0 if resp.status_code == 200 and not body.get("hasError") else 1)
