#!/usr/bin/env python3
"""
exp021: lambda_prior 0.6 → 0.5 (revert) + xSED [0.60,0.40] → [0.65,0.35]

Base: exp020 (lambda_prior=0.6, LB < 0.949)
Changes:
1. lambda_prior 0.6 → 0.5 at both call sites (revert to proven baseline)
2. xSED [0.60,0.40] → [0.65,0.35] in Cell 2 (solut dict) and Cell 13 (Model_5 dict)

Goal: test if more ProtoSSM weight (65%) improves over 0.949
"""
import json
from pathlib import Path

HERE    = Path(__file__).parent
NB_PATH = HERE / "birdclef-2026-eos-5.ipynb"

nb = json.loads(NB_PATH.read_text())
CHANGE_COUNT = 0

# ── Change 1&2: lambda_prior 0.6 → 0.5 (two call sites) ────────────────────
OLD_TE = 'hours=meta_te["hour_utc"].to_numpy(), tables=prior_tables, lambda_prior=0.6)'
NEW_TE = 'hours=meta_te["hour_utc"].to_numpy(), tables=prior_tables, lambda_prior=0.5)'

OLD_TR = 'hours=meta_tr["hour_utc"].to_numpy(), tables=prior_tables, lambda_prior=0.6)'
NEW_TR = 'hours=meta_tr["hour_utc"].to_numpy(), tables=prior_tables, lambda_prior=0.5)'

# ── Change 3: xSED [0.60,0.40] → [0.65,0.35] in Cell 2 solut dict ──────────
OLD_XSED_C2 = "  {'Model':'Model_5','subm':'subm_5.csv','weight':0.9673,'xSED':[0.60,0.40],'LB':'0.949'}"
NEW_XSED_C2 = "  {'Model':'Model_5','subm':'subm_5.csv','weight':0.9673,'xSED':[0.65,0.35],'LB':'0.949'}"

# ── Change 4: xSED [0.60,0.40] → [0.65,0.35] in Cell 13 Model_5 dict ───────
OLD_XSED_C13 = "         'xSED'  : [0.60,0.40],"
NEW_XSED_C13 = "         'xSED'  : [0.65,0.35],"

for cell in nb["cells"]:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

    if OLD_TE in src:
        src = src.replace(OLD_TE, NEW_TE)
        CHANGE_COUNT += 1
        print("✅ Change 1: lambda_prior 0.6→0.5 (test path)")

    if OLD_TR in src:
        src = src.replace(OLD_TR, NEW_TR)
        CHANGE_COUNT += 1
        print("✅ Change 2: lambda_prior 0.6→0.5 (train/OOF path)")

    if OLD_XSED_C2 in src:
        src = src.replace(OLD_XSED_C2, NEW_XSED_C2)
        CHANGE_COUNT += 1
        print("✅ Change 3: xSED [0.60,0.40]→[0.65,0.35] (Cell 2 solut dict)")

    if OLD_XSED_C13 in src:
        src = src.replace(OLD_XSED_C13, NEW_XSED_C13)
        CHANGE_COUNT += 1
        print("✅ Change 4: xSED [0.60,0.40]→[0.65,0.35] (Cell 13 Model_5 dict)")

    cell["source"] = src

if CHANGE_COUNT != 4:
    print(f"❌ Expected 4 changes, got {CHANGE_COUNT}. Aborting.")
    exit(1)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"\n✅ All {CHANGE_COUNT} changes applied → {NB_PATH}")
print("\nNext: python3 push_v9.py")
