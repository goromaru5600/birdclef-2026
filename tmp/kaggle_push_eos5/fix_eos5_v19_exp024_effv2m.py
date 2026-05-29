#!/usr/bin/env python3
"""
exp024: Add EfficientNetV2M ONNX (victorfernandezalbor/birdclef-2026-effv2m-ckpt)
        as additional blend component in Tucker SED stage

Architecture:
  p_mean = 0.85 * p_b0 (Tucker SED B0) + 0.15 * p_effv2m (EfficientNetV2M)
  → same xSED quantile-mix blend (0.60 Proto / 0.40 SED) uses enriched p_mean

Preprocessing (from af1981/ckpt-chain notebooks):
  librosa.power_to_db(mel, ref=np.max) → (x - mean) / (std + 1e-8)
  shape: (N, 128, 501, 1)  channels-last TF format

Changes:
1. kernel-metadata.json: add victorfernandezalbor/birdclef-2026-effv2m-ckpt
                              and beyondlogic/birdclef-2026-models-v3
2. Notebook Cell 13: setup code (after B1 SED setup)
3. Notebook Cell 13: inference blend (after gaussian_filter1d, per file)
"""
import json
from pathlib import Path

HERE    = Path(__file__).parent
META    = HERE / "kernel-metadata.json"
NB_PATH = HERE / "birdclef-2026-eos-5.ipynb"

# ── 1. kernel-metadata.json ──────────────────────────────────────────────────
meta = json.loads(META.read_text())
new_datasets = [
    "victorfernandezalbor/birdclef-2026-effv2m-ckpt",
    "beyondlogic/birdclef-2026-models-v3",
]
added = []
for ds in new_datasets:
    if ds not in meta["dataset_sources"]:
        meta["dataset_sources"].append(ds)
        added.append(ds)
META.write_text(json.dumps(meta, indent=2))
print(f"✅ kernel-metadata.json: added {added}")
print(f"   dataset_sources now ({len(meta['dataset_sources'])}): {meta['dataset_sources']}")

# ── 2. Notebook Cell 13 ───────────────────────────────────────────────────────
nb = json.loads(NB_PATH.read_text())
CHANGE_COUNT = 0

# ── Change 1: setup (after B1 SED print, before sed_rows) ────────────────────
OLD_SETUP = (
    '        print(f"B1 SED folds: {[p.name for p in b1_paths]}  weight={B1_WEIGHT}")\n'
    '\n'
    '        sed_rows, sed_preds = [], []'
)
NEW_SETUP = (
    '        print(f"B1 SED folds: {[p.name for p in b1_paths]}  weight={B1_WEIGHT}")\n'
    '\n'
    '        # ── EfficientNetV2M ONNX (victorfernandezalbor/birdclef-2026-effv2m-ckpt) ──\n'
    '        EFFV2M_W = 0.15\n'
    '        _effv2m_path = Path("/kaggle/input/birdclef-2026-effv2m-ckpt/effv2m_ckpt.onnx")\n'
    '        if _effv2m_path.exists():\n'
    '            effv2m_sess = make_sed_session(_effv2m_path)\n'
    '            effv2m_inp  = effv2m_sess.get_inputs()[0].name\n'
    '            with open("/kaggle/input/birdclef-2026-models-v3/ensemble_cfg.json") as _f:\n'
    '                _effv2m_col = np.array([label_to_idx.get(c, -1)\n'
    '                                        for c in json.load(_f)["primary_labels"]], dtype=np.int32)\n'
    '            def audio_to_mel_effv2m(chunks):\n'
    '                mels = []\n'
    '                for x in chunks:\n'
    '                    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=320,\n'
    '                                                        n_mels=128, fmin=20, fmax=16000, power=2.0)\n'
    '                    m = librosa.power_to_db(m, ref=np.max).astype(np.float32)\n'
    '                    mu, sg = m.mean(), m.std()\n'
    '                    m = (m - mu) / (sg + 1e-8)\n'
    '                    m = m[:, :501] if m.shape[1] >= 501 else np.pad(m, ((0,0),(0,501-m.shape[1])))\n'
    '                    mels.append(m)\n'
    '                return np.stack(mels)[:, :, :, np.newaxis].astype(np.float32)\n'
    '            print(f"EffV2M ONNX loaded  weight={EFFV2M_W}  matched={(_effv2m_col>=0).sum()}/234")\n'
    '        else:\n'
    '            effv2m_sess = None\n'
    '            print("EffV2M ONNX not found — B0-only")\n'
    '\n'
    '        sed_rows, sed_preds = [], []'
)

# ── Change 2: inference blend (after gaussian, before stem) ──────────────────
OLD_INFER = (
    '            if len(p_mean) > 1:\n'
    '                p_mean = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)\n'
    '        \n'
    '            stem = path.stem'
)
NEW_INFER = (
    '            if len(p_mean) > 1:\n'
    '                p_mean = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)\n'
    '\n'
    '            # ── EfficientNetV2M blend ─────────────────────────────────────\n'
    '            if effv2m_sess is not None:\n'
    '                _mel_e = audio_to_mel_effv2m(chunks)\n'
    '                _p_e = effv2m_sess.run(None, {effv2m_inp: _mel_e})[0]  # (N,234) sigmoid\n'
    '                _p_e_aligned = np.zeros_like(p_mean)\n'
    '                for _ei, _pi in enumerate(_effv2m_col):\n'
    '                    if _pi >= 0:\n'
    '                        _p_e_aligned[:, _pi] = _p_e[:, _ei]\n'
    '                p_mean = (1.0 - EFFV2M_W) * p_mean + EFFV2M_W * _p_e_aligned\n'
    '\n'
    '            stem = path.stem'
)

for cell in nb["cells"]:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

    if OLD_SETUP in src:
        src = src.replace(OLD_SETUP, NEW_SETUP)
        CHANGE_COUNT += 1
        print("✅ Change 1: EffV2M setup injected (after B1 SED)")

    if OLD_INFER in src:
        src = src.replace(OLD_INFER, NEW_INFER)
        CHANGE_COUNT += 1
        print("✅ Change 2: EffV2M inference blend injected (after gaussian)")

    cell["source"] = src

if CHANGE_COUNT != 2:
    print(f"❌ Expected 2 changes, got {CHANGE_COUNT}. Aborting.")
    exit(1)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"\n✅ All changes applied → {NB_PATH}")
print("\nNext: python3 push_v9.py")
