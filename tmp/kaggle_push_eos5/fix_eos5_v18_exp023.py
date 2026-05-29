#!/usr/bin/env python3
"""
exp023: rank_power 0.7→0.6 (revert) + file_confidence_scale power 0.4→0.3

Base: exp022 / v17 (rank_power=0.7, LB=0.948)
Changes:
1. rank_aware_scaling power=0.7 → 0.6 (revert to best known)
2. file_confidence_scale power=0.4 → 0.3 (new axis, less aggressive scaling)

Rationale: file_confidence_scale が強すぎると高信頼ファイルのスコアが
抑制される可能性がある。power=0.3 で緩和して改善を狙う。
"""
import json
from pathlib import Path

HERE    = Path(__file__).parent
NB_PATH = HERE / "birdclef-2026-eos-5.ipynb"

nb = json.loads(NB_PATH.read_text())
CHANGE_COUNT = 0

OLD_RANK = "probs = rank_aware_scaling(probs,    n_windows=N_WINDOWS, power=0.7)"
NEW_RANK = "probs = rank_aware_scaling(probs,    n_windows=N_WINDOWS, power=0.6)"

OLD_FCS  = "probs = file_confidence_scale(probs, n_windows=N_WINDOWS, top_k=2, power=0.4)"
NEW_FCS  = "probs = file_confidence_scale(probs, n_windows=N_WINDOWS, top_k=2, power=0.3)"

for cell in nb["cells"]:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

    if OLD_RANK in src:
        src = src.replace(OLD_RANK, NEW_RANK)
        CHANGE_COUNT += 1
        print("✅ Change 1: rank_aware_scaling power 0.7→0.6 (revert)")

    if OLD_FCS in src:
        src = src.replace(OLD_FCS, NEW_FCS)
        CHANGE_COUNT += 1
        print("✅ Change 2: file_confidence_scale power 0.4→0.3")

    cell["source"] = src

if CHANGE_COUNT != 2:
    print(f"❌ Expected 2 changes, got {CHANGE_COUNT}. Aborting.")
    exit(1)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"\n✅ All {CHANGE_COUNT} changes applied → {NB_PATH}")
print("\nNext: python3 push_v9.py")
