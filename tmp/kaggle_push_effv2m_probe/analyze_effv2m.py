#!/usr/bin/env python3
"""Download EffV2M probe output and test whether it escapes the 0.950 pile.

Decision criteria:
  - rank-corr(EoS.9 blend, EffV2M) << 0.99  -> genuinely decorrelated (good)
  - rank-mean(EoS.9, small*EffV2M) > EoS.9 alone -> blend helps (go)
If both fail, CNN-diversity thesis dead -> fall back to securing 0.950.
"""
import json, sys, requests
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tmp/kaggle_push_eos5"))
from oof_scorer import build_eval, macro_auc, rank_pct  # noqa

creds = json.loads(Path.home().joinpath(".kaggle/kaggle.json").read_text())
H = {"Authorization": f"Bearer {creds['key'].strip()}"}
U = creds["username"].strip()

# 1. download submission_effv2m.csv
out = HERE / "submission_effv2m.csv"
if not out.exists():
    d = requests.get(
        f"https://www.kaggle.com/api/v1/kernels/output?userName={U}&kernelSlug=birdclef2026-effv2m-probe",
        headers=H, timeout=60).json()
    url = next((f["url"] for f in d["files"] if f["fileName"] == "submission_effv2m.csv"), None)
    assert url, f"submission_effv2m.csv not in output: {[f['fileName'] for f in d['files']]}"
    out.write_bytes(requests.get(url, timeout=120).content)
    print("downloaded", out, out.stat().st_size, "bytes")

# 2. build aligned eval (EoS.9 proto/sed + EffV2M), intersect row_ids
e9 = ROOT / "tmp/kaggle_push_eos9_adopt/dryrun_out"
csvs = [("p9", str(e9 / "submission_protossm.csv")),
        ("s9", str(e9 / "submission_sed.csv")),
        ("eff", str(out))]
preds, Y, mask, rids, cols = build_eval(*csvs)
print(f"aligned rows={len(rids)}  evaluable classes={int(mask.sum())}")

B9 = 0.6 * rank_pct(preds["p9"]) + 0.4 * rank_pct(preds["s9"])
E = preds["eff"]
print("EoS.9 blend AUC :", round(macro_auc(B9, Y, mask)[0], 5))
print("EffV2M    AUC   :", round(macro_auc(E, Y, mask)[0], 5))

# non-degenerate check
nonconst = sum(1 for c in range(E.shape[1]) if mask[c] and E[:, c].std() > 1e-9)
print("EffV2M non-constant evaluable cols:", nonconst, "/", int(mask.sum()))

def col_corr(A, B):
    cs = []
    for c in range(A.shape[1]):
        if not mask[c] or A[:, c].std() == 0 or B[:, c].std() == 0:
            continue
        cs.append(np.corrcoef(rankdata(A[:, c]), rankdata(B[:, c]))[0, 1])
    return float(np.mean(cs)) if cs else float("nan")

print("rank-corr(EoS.9, EffV2M):", round(col_corr(B9, E), 4), " (lower = more diverse)")

B9r, Er = rank_pct(B9), rank_pct(E)
print("--- top-level rank blend ---")
base = macro_auc(B9, Y, mask)[0]
for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
    a = macro_auc((1 - w) * B9r + w * Er, Y, mask)[0]
    flag = "  <-- baseline" if w == 0 else ("  ↑" if a > base else "")
    print(f"  {1-w:.2f}*EoS9 + {w:.2f}*EffV2M : {a:.5f}{flag}")
