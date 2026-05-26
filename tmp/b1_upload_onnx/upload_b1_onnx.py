#!/usr/bin/env python3
"""Upload B1 pseudo-label ONNX files (sed_distill_fold0-4) to gorubachohu/560-sed-distill-fold0."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

src_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path("~/Downloads").expanduser()
onnx_files = [src_dir / f"sed_distill_fold{k}.onnx" for k in range(5)]

missing = [f for f in onnx_files if not f.exists()]
if missing:
    print(f"❌ Missing files: {[str(f) for f in missing]}")
    sys.exit(1)

print("✅ All 5 ONNX files found:")
for f in onnx_files:
    print(f"   {f}  ({f.stat().st_size/1e6:.1f} MB)")

upload_dir = Path(__file__).parent / "_upload_staging"
upload_dir.mkdir(exist_ok=True)

for f in onnx_files:
    shutil.copy2(f, upload_dir / f.name)

meta = {
    "title": "560 SED Distill Fold0",
    "id": "gorubachohu/560-sed-distill-fold0",
    "licenses": [{"name": "CC0-1.0"}]
}
(upload_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

print(f"\nUploading to gorubachohu/560-sed-distill-fold0 ...")
result = subprocess.run(
    [
        "/Users/goroishikura/.pyenv/versions/3.11.10/bin/kaggle",
        "datasets", "version",
        "-p", str(upload_dir),
        "-m", "Phase 3 pseudo-label B1 ONNX all folds 0-4",
        "--dir-mode", "zip"
    ],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    sys.exit(1)
print("✅ Upload complete!")
