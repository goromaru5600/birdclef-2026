#!/usr/bin/env python3
"""
exp025: file_confidence_scale 0.3→0.4 (revert) + EFFV2M_W 0.15→0.40

Problem found in v19:
  実際の EffV2M 寄与 = 0.40 (xSED SED weight) × 0.15 (EFFV2M_W) = 6% のみ
  さらに file_confidence=0.3 (-0.001) が残っていた

Fix:
1. file_confidence_scale power=0.3 → 0.4 (baseline に戻す)
2. EFFV2M_W=0.15 → 0.40 (最終ブレンドで 0.40×0.40=16% の寄与)

Final blend:
  0.60 × ProtoSSM + 0.40 × (0.60×Tucker + 0.40×EffV2M)
= 0.60 × ProtoSSM + 0.24 × Tucker + 0.16 × EffV2M
"""
import json
from pathlib import Path

HERE    = Path(__file__).parent
NB_PATH = HERE / "birdclef-2026-eos-5.ipynb"

nb = json.loads(NB_PATH.read_text())
CHANGE_COUNT = 0

OLD_FCS  = "probs = file_confidence_scale(probs, n_windows=N_WINDOWS, top_k=2, power=0.3)"
NEW_FCS  = "probs = file_confidence_scale(probs, n_windows=N_WINDOWS, top_k=2, power=0.4)"

OLD_W    = "        EFFV2M_W = 0.15"
NEW_W    = "        EFFV2M_W = 0.40"

for cell in nb["cells"]:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

    if OLD_FCS in src:
        src = src.replace(OLD_FCS, NEW_FCS)
        CHANGE_COUNT += 1
        print("✅ Change 1: file_confidence_scale power 0.3→0.4 (revert)")

    if OLD_W in src:
        src = src.replace(OLD_W, NEW_W)
        CHANGE_COUNT += 1
        print("✅ Change 2: EFFV2M_W 0.15→0.40")

    cell["source"] = src

if CHANGE_COUNT != 2:
    print(f"❌ Expected 2 changes, got {CHANGE_COUNT}. Aborting.")
    exit(1)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"\n✅ All {CHANGE_COUNT} changes applied → {NB_PATH}")
print("\nNext: python3 push_v9.py")
