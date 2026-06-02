#!/usr/bin/env python3
"""Gate test: does the soundscape-distilled B0 help the EoS.9 blend?

Downloads val_preds.csv (B0 preds on the 66 labeled soundscape files) and tests,
on the oof_scorer harness (EoS.9 proto/sed, 190 rows / 42 classes):
  - B0 standalone macro-AUC
  - rank-corr(EoS.9 blend, B0)        (low = diverse, like EffV2M's 0.31)
  - rank-blend(EoS.9, w*B0) vs EoS.9   (does a small weight beat 0.950 baseline?)
"""
import json, sys, time, requests
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

out = HERE / "val_preds.csv"
if not out.exists() or out.stat().st_size < 50000:
    d = requests.get(f"https://www.kaggle.com/api/v1/kernels/output?userName={U}&kernelSlug=birdclef2026-train-ssc-b0",
                     headers=H, timeout=60).json()
    url = next(f["url"] for f in d["files"] if f["fileName"] == "val_preds.csv")
    for attempt in range(5):
        try:
            with requests.get(url, timeout=180, stream=True) as r:
                r.raise_for_status()
                with open(out, "wb") as fh:
                    for ch in r.iter_content(65536):
                        fh.write(ch)
            break
        except Exception as e:
            print("retry", attempt, str(e)[:60]); time.sleep(3)
    print("downloaded", out.stat().st_size, "bytes")

e9 = ROOT / "tmp/kaggle_push_eos9_adopt/dryrun_out"
csvs = [("p9", str(e9 / "submission_protossm.csv")),
        ("s9", str(e9 / "submission_sed.csv")),
        ("b0", str(out))]
preds, Y, mask, rids, cols = build_eval(*csvs)
print(f"aligned rows={len(rids)}  evaluable classes={int(mask.sum())}")

B9 = 0.6 * rank_pct(preds["p9"]) + 0.4 * rank_pct(preds["s9"])
B0 = preds["b0"]
print("EoS.9 blend AUC :", round(macro_auc(B9, Y, mask)[0], 5))
print("B0 (ssc) AUC    :", round(macro_auc(B0, Y, mask)[0], 5))
print("B0 non-const cols:", sum(1 for c in range(B0.shape[1]) if mask[c] and B0[:, c].std() > 1e-9), "/", int(mask.sum()))

def col_corr(A, B):
    cs = [np.corrcoef(rankdata(A[:, c]), rankdata(B[:, c]))[0, 1]
          for c in range(A.shape[1]) if mask[c] and A[:, c].std() > 0 and B[:, c].std() > 0]
    return float(np.mean(cs)) if cs else float("nan")

print("rank-corr(EoS.9, B0):", round(col_corr(B9, B0), 4), " (lower=more diverse; EffV2M was 0.31)")

B9r, B0r = rank_pct(B9), rank_pct(B0)
base = macro_auc(B9, Y, mask)[0]
print("--- top-level rank blend (EoS.9 + w*B0) ---")
best = (base, 0.0)
for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    a = macro_auc((1 - w) * B9r + w * B0r, Y, mask)[0]
    if a > best[0]:
        best = (a, w)
    print(f"  w={w:.2f}: {a:.5f}{'  <-- baseline' if w==0 else ('  UP' if a>base else '')}")
print(f"BEST blend: AUC {best[0]:.5f} at w={best[1]:.2f}  (delta vs EoS.9 {best[0]-base:+.5f})")
