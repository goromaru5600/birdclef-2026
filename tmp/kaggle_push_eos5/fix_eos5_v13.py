#!/usr/bin/env python3
"""EoS5 v13: restore B1 SED code (Phase 3 pseudo-label, all 5 folds)."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "birdclef-2026-eos-5.ipynb"
nb = json.loads(NB_PATH.read_text())

CHANGE_COUNT = 0

for cell in nb["cells"]:
    src = cell["source"]
    if isinstance(src, list):
        src = "".join(src)

    # ── Change 1: Add B1 setup after B0 sessions print ──────────────────────
    OLD1 = (
        '        sed_sessions_b0 = [make_sed_session(p) for p in sed_fold_paths]\n'
        '        print(f"B0 SED folds: {[p.name for p in sed_fold_paths]}")\n'
        '\n'
        '\n'
        '        sed_rows, sed_preds = [], []'
    )
    NEW1 = (
        '        sed_sessions_b0 = [make_sed_session(p) for p in sed_fold_paths]\n'
        '        print(f"B0 SED folds: {[p.name for p in sed_fold_paths]}")\n'
        '\n'
        '        # ── B1 SED (Phase 3 pseudo-label retrained, all 5 folds, 128 mel) ────────────\n'
        '        def audio_to_mel_128(chunks):\n'
        '            mels = []\n'
        '            for x in chunks:\n'
        '                s = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,\n'
        '                                                    n_mels=128, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0)\n'
        '                s = librosa.power_to_db(s, top_db=TOP_DB_SED)\n'
        '                s = (s - s.mean()) / (s.std() + 1e-6)\n'
        '                mels.append(s)\n'
        '            return np.stack(mels)[:, None].astype(np.float32)\n'
        '\n'
        '        B1_WEIGHT = 0.10  # Phase 3: pseudo-label retrained, all 5 folds\n'
        '        b1_dir = Path("/kaggle/input/560-sed-distill-fold0")\n'
        '        b1_paths = [b1_dir / f"fold{k}.onnx" for k in range(5)\n'
        '                    if (b1_dir / f"fold{k}.onnx").exists()]\n'
        '        sed_sessions_b1 = [make_sed_session(p) for p in b1_paths]\n'
        '        print(f"B1 SED folds: {[p.name for p in b1_paths]}  weight={B1_WEIGHT}")\n'
        '\n'
        '        sed_rows, sed_preds = [], []'
    )
    if OLD1 in src:
        src = src.replace(OLD1, NEW1)
        CHANGE_COUNT += 1
        print("✅ Change 1: B1 setup added")

    # ── Change 2: Add mel_128 computation inside loop ────────────────────────
    OLD2 = (
        '        for i, path in enumerate(test_paths, 1):\n'
        '            chunks, ends = file_to_sed_chunks(path)\n'
        '            mel_256 = audio_to_mel(chunks)\n'
        '\n'
        '            # B0 予測'
    )
    NEW2 = (
        '        for i, path in enumerate(test_paths, 1):\n'
        '            chunks, ends = file_to_sed_chunks(path)\n'
        '            mel_256 = audio_to_mel(chunks)\n'
        '            mel_128 = audio_to_mel_128(chunks) if sed_sessions_b1 else None\n'
        '\n'
        '            # B0 予測'
    )
    if OLD2 in src:
        src = src.replace(OLD2, NEW2)
        CHANGE_COUNT += 1
        print("✅ Change 2: mel_128 added inside loop")

    # ── Change 3: Replace p_mean = p_b0 with B1 blend ───────────────────────
    OLD3 = (
        '            p_b0 /= len(sed_sessions_b0)\n'
        '            p_mean = p_b0'
    )
    NEW3 = (
        '            p_b0 /= len(sed_sessions_b0)\n'
        '\n'
        '            # B1 予測 (Phase 3 pseudo-label retrained all folds)\n'
        '            if sed_sessions_b1:\n'
        '                p_b1 = np.zeros((len(chunks), N_CLASSES), dtype=np.float32)\n'
        '                for sess in sed_sessions_b1:\n'
        '                    outs = sess.run(None, {sess.get_inputs()[0].name: mel_128})\n'
        '                    p_b1 += 0.5 * sigmoid_sed(outs[0]) + 0.5 * sigmoid_sed(outs[1].max(axis=1))\n'
        '                p_b1 /= len(sed_sessions_b1)\n'
        '                p_mean = (1.0 - B1_WEIGHT) * p_b0 + B1_WEIGHT * p_b1\n'
        '            else:\n'
        '                p_mean = p_b0'
    )
    if OLD3 in src:
        src = src.replace(OLD3, NEW3)
        CHANGE_COUNT += 1
        print("✅ Change 3: B1 blend code added")

    cell["source"] = src

if CHANGE_COUNT != 3:
    print(f"❌ Expected 3 changes, got {CHANGE_COUNT}. Check strings carefully.")
    exit(1)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"\n✅ All {CHANGE_COUNT} changes applied → {NB_PATH}")
