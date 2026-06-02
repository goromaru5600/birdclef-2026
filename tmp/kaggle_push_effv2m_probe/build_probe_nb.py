#!/usr/bin/env python3
"""Build a self-contained EffV2M dry-run probe notebook.

Dumps EffV2M standalone predictions over ALL train_soundscapes files,
row_id = {stem}_{end_sec} (matching EoS.9 dry-run convention), aligned to the
234 competition columns. Output: /kaggle/working/submission_effv2m.csv

Purpose: download this + measure rank-correlation vs EoS.9 and test whether a
top-level rank blend escapes the 0.950 pile.  No competition submission consumed.
"""
import json
from pathlib import Path

CODE = r'''
import os, sys, json, glob, subprocess
import numpy as np, pandas as pd
from pathlib import Path

# --- onnxruntime (use base image if present, else install from attached wheel, no internet) ---
try:
    import onnxruntime as ort
except Exception:
    whls = glob.glob('/kaggle/input/**/onnxruntime-*.whl', recursive=True)
    assert whls, "onnxruntime wheel not found in inputs"
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', whls[0]], check=True)
    import onnxruntime as ort
import librosa
print("onnxruntime", ort.__version__)

SR = 32000

# --- competition paths ---
COMP = Path('/kaggle/input/birdclef-2026')
if not COMP.exists():
    COMP = Path('/kaggle/input/competitions/birdclef-2026')
sample = pd.read_csv(COMP / 'sample_submission.csv')
LABELS = sample.columns[1:].tolist()
label_to_idx = {c: i for i, c in enumerate(LABELS)}
print("classes:", len(LABELS))

# --- EffV2M ONNX ---
effp = glob.glob('/kaggle/input/**/effv2m_ckpt.onnx', recursive=True)
assert effp, "effv2m_ckpt.onnx not found"
sess = ort.InferenceSession(effp[0], providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0].name
print("EffV2M input:", sess.get_inputs()[0].shape)

# --- label alignment via ensemble_cfg.json primary_labels ---
cfgp = glob.glob('/kaggle/input/**/ensemble_cfg.json', recursive=True)
assert cfgp, "ensemble_cfg.json not found"
prim = json.load(open(cfgp[0]))['primary_labels']
col = np.array([label_to_idx.get(c, -1) for c in prim], dtype=np.int32)
print("matched columns:", int((col >= 0).sum()), "/", len(prim))

def to_mel(x):
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=320,
                                       n_mels=128, fmin=20, fmax=16000, power=2.0)
    m = librosa.power_to_db(m, ref=np.max).astype(np.float32)
    mu, sg = m.mean(), m.std()
    m = (m - mu) / (sg + 1e-8)
    if m.shape[1] >= 501:
        m = m[:, :501]
    else:
        m = np.pad(m, ((0, 0), (0, 501 - m.shape[1])))
    return m

files = sorted((COMP / 'train_soundscapes').glob('*.ogg'))
print("train_soundscapes files:", len(files))

rows, preds = [], []
for fi, f in enumerate(files):
    try:
        y, _ = librosa.load(str(f), sr=SR)
    except Exception as e:
        print("load fail", f.name, e); continue
    stem = f.stem
    mels, ends = [], []
    for end in range(5, 65, 5):
        seg = y[(end - 5) * SR: end * SR]
        if len(seg) < SR:
            continue
        if len(seg) < 5 * SR:
            seg = np.pad(seg, (0, 5 * SR - len(seg)))
        mels.append(to_mel(seg)); ends.append(end)
    if not mels:
        continue
    X = np.stack(mels)[:, :, :, None].astype(np.float32)
    pe = sess.run(None, {inp: X})[0]
    al = np.zeros((len(mels), len(LABELS)), np.float32)
    for ei, pi in enumerate(col):
        if pi >= 0:
            al[:, pi] = pe[:, ei]
    for j, end in enumerate(ends):
        rows.append(f"{stem}_{end}"); preds.append(al[j])
    if fi % 10 == 0:
        print(f"  {fi+1}/{len(files)} files done")

df = pd.DataFrame(preds, columns=LABELS)
df.insert(0, 'row_id', rows)
df.to_csv('/kaggle/working/submission_effv2m.csv', index=False)
print("WROTE submission_effv2m.csv", df.shape)
'''

nb = {
    "cells": [
        {"cell_type": "code", "metadata": {}, "execution_count": None,
         "outputs": [], "source": CODE.strip("\n")}
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = Path(__file__).parent / "effv2m-probe.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", out)
