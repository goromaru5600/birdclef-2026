#!/usr/bin/env python3
"""
exp020: lambda_prior 0.5 → 0.6 (B0-only, no B1)

Changes:
1. kernel-metadata.json: remove gorubachohu/560-sed-distill-fold0
   (B1 ONNX won't be found → sed_sessions_b1=[] → p_mean = p_b0)
2. notebook Cell 13: lambda_prior=0.5 → 0.6 at both actual call sites (not comments)

Base: v12/exp019 (B0-only, LB=0.949)
Goal: test if lambda_prior=0.6 improves on 0.5
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
META_PATH = HERE / "kernel-metadata.json"
NB_PATH   = HERE / "birdclef-2026-eos-5.ipynb"

# ── 1. kernel-metadata.json ──────────────────────────────────────────────────
meta = json.loads(META_PATH.read_text())
before = list(meta["dataset_sources"])
meta["dataset_sources"] = [s for s in meta["dataset_sources"]
                            if s != "gorubachohu/560-sed-distill-fold0"]
after = meta["dataset_sources"]

removed = set(before) - set(after)
if removed:
    print(f"✅ Removed from dataset_sources: {removed}")
else:
    print("ℹ️  gorubachohu/560-sed-distill-fold0 was not in dataset_sources (already clean)")

META_PATH.write_text(json.dumps(meta, indent=2))
print(f"   dataset_sources now: {meta['dataset_sources']}")

# ── 2. notebook: lambda_prior 0.5 → 0.6 ────────────────────────────────────
nb = json.loads(NB_PATH.read_text())
CHANGE_COUNT = 0

# Exact strings to replace (non-comment call sites only)
OLD_TE = 'hours=meta_te["hour_utc"].to_numpy(), tables=prior_tables, lambda_prior=0.5)'
NEW_TE = 'hours=meta_te["hour_utc"].to_numpy(), tables=prior_tables, lambda_prior=0.6)'

OLD_TR = 'hours=meta_tr["hour_utc"].to_numpy(), tables=prior_tables, lambda_prior=0.5)'
NEW_TR = 'hours=meta_tr["hour_utc"].to_numpy(), tables=prior_tables, lambda_prior=0.6)'

for cell in nb["cells"]:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

    if OLD_TE in src:
        src = src.replace(OLD_TE, NEW_TE)
        CHANGE_COUNT += 1
        print("✅ Change 1: lambda_prior=0.6 (test path)")

    if OLD_TR in src:
        src = src.replace(OLD_TR, NEW_TR)
        CHANGE_COUNT += 1
        print("✅ Change 2: lambda_prior=0.6 (train/OOF calibration path)")

    cell["source"] = src

if CHANGE_COUNT != 2:
    print(f"❌ Expected 2 changes, got {CHANGE_COUNT}. Aborting.")
    exit(1)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"\n✅ All changes applied → {NB_PATH}")
print("\nNext: python push_v9.py")
