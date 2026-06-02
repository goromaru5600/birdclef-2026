#!/usr/bin/env python3
"""Push the EffV2M probe notebook (Bearer auth, KGAT_ token)."""
import json, sys
from pathlib import Path
import requests

HERE = Path(__file__).parent
meta = json.loads((HERE / "kernel-metadata.json").read_text())
nb_json = json.loads((HERE / "effv2m-probe.ipynb").read_text())

creds = json.loads(Path.home().joinpath(".kaggle/kaggle.json").read_text())
KEY = creds["key"].strip()
assert KEY.startswith("KGAT_"), "Expected KGAT_ token"

for cell in nb_json.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
    if isinstance(cell.get("source"), list):
        cell["source"] = "".join(cell["source"])
nb_text = json.dumps(nb_json)

payload = {
    "slug": meta["id"],
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
headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "Accept": "application/json"}
print(f"Pushing {meta['id']} ({len(nb_text)} chars)...")
resp = requests.post("https://www.kaggle.com/api/v1/kernels/push", headers=headers, json=payload, timeout=300)
print("Status:", resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
except Exception:
    print(resp.text)
body = resp.json() if resp.status_code == 200 else {}
sys.exit(0 if resp.status_code == 200 and not body.get("hasError") else 1)
