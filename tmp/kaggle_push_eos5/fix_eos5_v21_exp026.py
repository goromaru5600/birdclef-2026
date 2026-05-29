#!/usr/bin/env python3
"""
exp026: disable EffV2M (it hurt: 0.948@6% -> 0.947@16%) + shift xSED toward SED.

Rationale (OOF validation, oof_scorer.py):
  - production already quantile/rank-blends proto+sed (cell 13 _qp/_qs)
  - on validation, AUC rises MONOTONICALLY as SED weight increases
      current 0.60proto/0.40sed = 0.99194  ->  0.50/0.50 = 0.99353
  - every prior LB xSED experiment moved toward MORE proto (0.65) and all
    dropped to 0.948. The MORE-SED direction has never been tried on LB.

Changes (from current v20 notebook state):
  1. EFFV2M_W 0.40 -> 0.0   (cleanly disable EffV2M, back to exp019 baseline)
  2. xSED [0.60,0.40] -> [0.50,0.50] in Cell 2 (solut) and Cell 13 (Model_5)

Net: exp019 (0.949, no EffV2M) with the single isolated change of SED weight up.
"""
import json
from pathlib import Path

HERE    = Path(__file__).parent
NB_PATH = HERE / "birdclef-2026-eos-5.ipynb"

nb = json.loads(NB_PATH.read_text())
CHANGE_COUNT = 0

OLD_W  = "        EFFV2M_W = 0.40"
NEW_W  = "        EFFV2M_W = 0.0"

OLD_C2 = "  {'Model':'Model_5','subm':'subm_5.csv','weight':0.9673,'xSED':[0.60,0.40],'LB':'0.949'}"
NEW_C2 = "  {'Model':'Model_5','subm':'subm_5.csv','weight':0.9673,'xSED':[0.50,0.50],'LB':'0.949'}"

OLD_C13 = "         'xSED'  : [0.60,0.40],"
NEW_C13 = "         'xSED'  : [0.50,0.50],"

for cell in nb["cells"]:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

    if OLD_W in src:
        src = src.replace(OLD_W, NEW_W)
        CHANGE_COUNT += 1
        print("✅ Change 1: EFFV2M_W 0.40 → 0.0 (disable EffV2M)")

    if OLD_C2 in src:
        src = src.replace(OLD_C2, NEW_C2)
        CHANGE_COUNT += 1
        print("✅ Change 2: xSED [0.60,0.40]→[0.50,0.50] (Cell 2 solut)")

    if OLD_C13 in src:
        src = src.replace(OLD_C13, NEW_C13)
        CHANGE_COUNT += 1
        print("✅ Change 3: xSED [0.60,0.40]→[0.50,0.50] (Cell 13 Model_5)")

    cell["source"] = src

if CHANGE_COUNT != 3:
    print(f"❌ Expected 3 changes, got {CHANGE_COUNT}. Aborting.")
    exit(1)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"\n✅ All {CHANGE_COUNT} changes applied → {NB_PATH}")
print("\nNext: python3 push_v9.py")
