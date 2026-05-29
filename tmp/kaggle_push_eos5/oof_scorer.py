#!/usr/bin/env python3
"""
Local OOF scorer for BirdCLEF-2026 EoS5 pipeline.

Reconstructs the validation set from dry-run submission CSVs
(train_soundscapes, 20 files x 12 windows -> 190 labeled rows) joined with
train_soundscapes_labels.csv.  Lets us tune blend weights / rank-vs-linear /
extra models OFFLINE without spending Kaggle submissions.

eval_data.npz (P,S,Y,class_mask, 190 rows / 42 evaluable classes) was produced
from exactly this dry run.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]   # birdclef-2026/
LABELS = ROOT / "data/train_soundscapes_labels.csv"


def _end_sec(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def load_truth(class_cols):
    lab = pd.read_csv(LABELS)
    lab["rid"] = (lab.filename.str.replace(".ogg", "", regex=False)
                  + "_" + lab.end.map(_end_sec).astype(str))
    col_idx = {c: i for i, c in enumerate(class_cols)}
    Y = {}
    for rid, sub in lab.groupby("rid"):
        v = np.zeros(len(class_cols), np.float32)
        for s in sub.primary_label.dropna():
            for t in str(s).split(";"):
                t = t.strip()
                if t in col_idx:
                    v[col_idx[t]] = 1.0
        Y[rid] = v
    return Y


def load_preds(csv_path, ref_rids=None):
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c != "row_id"]
    df = df.set_index("row_id")
    if ref_rids is not None:
        df = df.loc[ref_rids]
    return df[cols].to_numpy(np.float32), cols


def build_eval(*csv_paths):
    """Returns dict name->pred(np), Y(np), class_mask(np), rids(list), cols."""
    df0 = pd.read_csv(csv_paths[0][1])
    cols = [c for c in df0.columns if c != "row_id"]
    Y_map = load_truth(cols)
    rids = [r for r in df0.row_id if r in Y_map]          # 190 labeled rows
    Y = np.stack([Y_map[r] for r in rids])
    preds = {}
    for name, path in csv_paths:
        preds[name], _ = load_preds(path, ref_rids=rids)
    mask = (Y.sum(0) > 0) & (Y.sum(0) < len(rids))         # both pos+neg present
    return preds, Y, mask, rids, cols


def macro_auc(pred, Y, mask):
    aucs = []
    for c in range(Y.shape[1]):
        if not mask[c]:
            continue
        yc = Y[:, c]
        if yc.min() == yc.max():
            continue
        aucs.append(roc_auc_score(yc, pred[:, c]))
    return float(np.mean(aucs)), len(aucs)


def rank_pct(x):
    r = np.empty_like(x, dtype=np.float64)
    for c in range(x.shape[1]):
        r[:, c] = rankdata(x[:, c]) / (x.shape[0] + 1)
    return r


if __name__ == "__main__":
    HERE = Path(__file__).parent
    csvs = [
        ("proto",   "tmp/kaggle_push_946_fork/submission_protossm.csv"),
        ("sed",     "tmp/kaggle_push_946_fork/submission_sed.csv"),
        ("birdnet", "tmp/kaggle_push_946_fork/submission_birdnet.csv"),
    ]
    csvs = [(n, str(ROOT / p)) for n, p in csvs]
    preds, Y, mask, rids, cols = build_eval(*csvs)
    print(f"rows={len(rids)}  evaluable classes={mask.sum()}")

    # --- reproduce eval_data.npz ---
    ev = np.load(HERE / "eval_data.npz")
    print("\n[reproduce eval_data.npz]")
    print("  P match:", np.allclose(preds["proto"], ev["P"], atol=1e-4),
          " S match:", np.allclose(preds["sed"], ev["S"], atol=1e-4),
          " Y match:", np.array_equal(Y, ev["Y"]),
          " mask match:", np.array_equal(mask, ev["class_mask"]))

    P, S, B = preds["proto"], preds["sed"], preds["birdnet"]
    print("\n[singles]")
    for n, p in [("proto", P), ("sed", S), ("birdnet", B)]:
        print(f"  {n:8s}", macro_auc(p, Y, mask))

    Pr, Sr, Br = rank_pct(P), rank_pct(S), rank_pct(B)
    print("\n[proto+sed  linear vs rank]")
    for w in (0.4, 0.5, 0.6, 0.7):
        print(f"  w_proto={w:.1f}  lin={macro_auc(w*P+(1-w)*S,Y,mask)[0]:.5f}"
              f"  rank={macro_auc(w*Pr+(1-w)*Sr,Y,mask)[0]:.5f}")

    print("\n[+birdnet rank 3-way]  (proto,sed,birdnet)")
    best = (0, None)
    for wp in (0.2, 0.3, 0.4, 0.5):
        for ws in (0.2, 0.3, 0.4, 0.5):
            wb = 1 - wp - ws
            if wb < 0 or wb > 0.6:
                continue
            a = macro_auc(wp*Pr + ws*Sr + wb*Br, Y, mask)[0]
            if a > best[0]:
                best = (a, (wp, ws, round(wb, 2)))
    print("  best rank 3-way:", best)
