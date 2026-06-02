import json, sys
from pathlib import Path
import requests
HERE=Path(__file__).parent
meta=json.loads((HERE/"kernel-metadata.json").read_text())
nb=json.loads((HERE/"train-ssc-b0-v2.ipynb").read_text())
creds=json.loads(Path.home().joinpath(".kaggle/kaggle.json").read_text())
KEY=creds["key"].strip(); assert KEY.startswith("KGAT_")
for c in nb.get("cells",[]):
    if c.get("cell_type")=="code": c["outputs"]=[]
    if isinstance(c.get("source"),list): c["source"]="".join(c["source"])
payload={"slug":meta["id"],"newTitle":meta["title"],"text":json.dumps(nb),
 "language":meta["language"],"kernelType":meta["kernel_type"],"isPrivate":meta["is_private"],
 "enableGpu":meta["enable_gpu"],"enableTpu":meta["enable_tpu"],"enableInternet":meta["enable_internet"],
 "datasetDataSources":meta["dataset_sources"],"competitionDataSources":meta["competition_sources"],
 "kernelDataSources":meta["kernel_sources"],"modelDataSources":meta["model_sources"],"categoryIds":[]}
h={"Authorization":f"Bearer {KEY}","Content-Type":"application/json","Accept":"application/json"}
r=requests.post("https://www.kaggle.com/api/v1/kernels/push",headers=h,json=payload,timeout=300)
print("Status",r.status_code); print(json.dumps(r.json(),indent=1,ensure_ascii=False))
sys.exit(0 if r.status_code==200 and not r.json().get("hasError") else 1)
