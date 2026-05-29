#!/usr/bin/env python3
"""
exp022: xSED [0.65,0.35] → [0.60,0.40] (revert) + rank_aware_scaling power 0.6 → 0.7

Base: exp021 / v16 (xSED=[0.65,0.35], lambda_prior=0.5, LB=0.948)
Changes:
1. xSED [0.65,0.35] → [0.60,0.40] in Cell 2 (solut dict) — revert to best known
2. xSED [0.65,0.35] → [0.60,0.40] in Cell 13 (Model_5 dict) — revert to best known
3. rank_aware_scaling power=0.6 → 0.7 in Cell 13

History:
  exp019: power 0.5→0.6 → LB 0.948→0.949 (+0.001)  ← continues this direction
  exp022: power 0.6→0.7 → target > 0.949
"""
import json
from pathlib import Path

HERE    = Path(__file__).parent
NB_PATH = HERE / "birdclef-2026-eos-5.ipynb"

nb = json.loads(NB_PATH.read_text())
CHANGE_COUNT = 0

# ── Change 1: xSED revert in Cell 2 ─────────────────────────────────────────
OLD_XSED_C2 = "  {'Model':'Model_5','subm':'subm_5.csv','weight':0.9673,'xSED':[0.65,0.35],'LB':'0.949'}"
NEW_XSED_C2 = "  {'Model':'Model_5','subm':'subm_5.csv','weight':0.9673,'xSED':[0.60,0.40],'LB':'0.949'}"

# ── Change 2: xSED revert in Cell 13 ────────────────────────────────────────
OLD_XSED_C13 = "         'xSED'  : [0.65,0.35],"
NEW_XSED_C13 = "         'xSED'  : [0.60,0.40],"

# ── Change 3: rank_aware_scaling power 0.6 → 0.7 ────────────────────────────
OLD_RANK = "probs = rank_aware_scaling(probs,    n_windows=N_WINDOWS, power=0.6)"
NEW_RANK = "probs = rank_aware_scaling(probs,    n_windows=N_WINDOWS, power=0.7)"

for cell in nb["cells"]:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

    if OLD_XSED_C2 in src:
        src = src.replace(OLD_XSED_C2, NEW_XSED_C2)
        CHANGE_COUNT += 1
        print("✅ Change 1: xSED [0.65,0.35]→[0.60,0.40] (Cell 2)")

    if OLD_XSED_C13 in src:
        src = src.replace(OLD_XSED_C13, NEW_XSED_C13)
        CHANGE_COUNT += 1
        print("✅ Change 2: xSED [0.65,0.35]→[0.60,0.40] (Cell 13)")

    if OLD_RANK in src:
        src = src.replace(OLD_RANK, NEW_RANK)
        CHANGE_COUNT += 1
        print("✅ Change 3: rank_aware_scaling power 0.6→0.7")

    cell["source"] = src

if CHANGE_COUNT != 3:
    print(f"❌ Expected 3 changes, got {CHANGE_COUNT}. Aborting.")
    exit(1)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"\n✅ All {CHANGE_COUNT} changes applied → {NB_PATH}")
print("\nNext: python3 push_v9.py")
