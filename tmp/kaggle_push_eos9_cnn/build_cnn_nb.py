#!/usr/bin/env python3
"""Build EoS.9 + ssc-B0(v2) top-level rank-blend notebook (real-LB test).

Does NOT touch EoS.9 internals. Appends 2 cells AFTER the final submission.csv:
  Cell +1: run ssc_b0_v2.onnx over test_soundscapes -> /kaggle/working/subm_cnn.csv
  Cell +2: rank-blend submission.csv (EoS.9, row_id index) with subm_cnn at w=0.05

0.950 base kernel is untouched (separate slug). If CNN file missing or run fails
mid-way, the cells are guarded so the EoS.9 submission.csv (0.950) survives.
"""
import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "kaggle_push_eos9_adopt/birdclef-2026-eos-9.ipynb"
nb = json.loads(SRC.read_text())

CNN_CELL = r'''
# ===== ssc-B0(v2) CNN inference for top-level rank blend =====
import numpy as _np, pandas as _pd, glob as _glob, os as _os, librosa as _lb
import onnxruntime as _ort
from pathlib import Path as _P
_SR=32000
_mp=_glob.glob('/kaggle/input/**/ssc_b0_v2.onnx', recursive=True)
print('CNN model path:', _mp)
if _mp:
    _so=_ort.SessionOptions(); _so.intra_op_num_threads=4
    _cnn=_ort.InferenceSession(_mp[0], sess_options=_so, providers=['CPUExecutionProvider'])
    _cin=_cnn.get_inputs()[0].name
    _BASE=_P('/kaggle/input/birdclef-2026')
    if not _BASE.exists(): _BASE=_P('/kaggle/input/competitions/birdclef-2026')
    _tp=sorted((_BASE/'test_soundscapes').glob('*.ogg'))
    if len(_tp)==0: _tp=sorted((_BASE/'train_soundscapes').glob('*.ogg'))[:40]
    print('CNN over', len(_tp), 'files')
    _LB=_pd.read_csv(_BASE/'sample_submission.csv').columns[1:].tolist()
    def _mel(x):
        m=_lb.feature.melspectrogram(y=x,sr=_SR,n_fft=1024,hop_length=320,n_mels=128,fmin=20,fmax=16000,power=2.0)
        m=_lb.power_to_db(m,ref=_np.max).astype(_np.float32); m=(m-m.mean())/(m.std()+1e-8)
        return m[:,:501] if m.shape[1]>=501 else _np.pad(m,((0,0),(0,501-m.shape[1])))
    _rows=[]; _prs=[]
    for _f in _tp:
        try: _y,_=_lb.load(str(_f),sr=_SR)
        except Exception: continue
        _st=_f.stem; _ml=[]; _en=[]
        for _e in range(5,65,5):
            _s=_y[(_e-5)*_SR:_e*_SR]
            if len(_s)<_SR: continue
            if len(_s)<5*_SR: _s=_np.pad(_s,(0,5*_SR-len(_s)))
            _ml.append(_mel(_s)); _en.append(_e)
        if not _ml: continue
        _X=_np.stack(_ml)[:,None,:,:].astype(_np.float32)
        _lg=_cnn.run(None,{_cin:_X})[0]
        _p=1.0/(1.0+_np.exp(-_lg))
        for _j,_e in enumerate(_en): _rows.append(f'{_st}_{_e}'); _prs.append(_p[_j])
    _cdf=_pd.DataFrame(_prs,columns=_LB); _cdf.insert(0,'row_id',_rows)
    _cdf.to_csv('/kaggle/working/subm_cnn.csv', index=False)
    print('subm_cnn.csv', _cdf.shape)
else:
    print('ssc_b0_v2.onnx NOT found - skipping CNN (submission stays pure EoS.9)')
'''

BLEND_CELL = r'''
# ===== rank-blend final submission.csv with CNN (w=0.05) =====
import pandas as _pd, numpy as _np, os as _os
_W=0.05
if _os.path.exists('/kaggle/working/subm_cnn.csv') and _os.path.exists('submission.csv'):
    _sub=_pd.read_csv('submission.csv', index_col=0)
    _cnn=_pd.read_csv('/kaggle/working/subm_cnn.csv').set_index('row_id')
    _cols=list(_sub.columns)
    _cnn=_cnn.reindex(_sub.index)[_cols]
    _er=_sub.rank(axis=0, pct=True)
    _cr=_cnn.rank(axis=0, pct=True)
    _cr=_cr.where(~_cr.isna(), _er)             # rows w/o CNN -> EoS.9 rank
    _bl=(1.0-_W)*_er.values + _W*_cr.values
    _out=_pd.DataFrame(_bl.astype('float32'), index=_sub.index, columns=_cols)
    _out.index.name=_sub.index.name
    _out.to_csv('submission.csv', index=True)
    print(f'BLENDED submission.csv with CNN w={_W}  shape={_out.shape}  mean={_out.values.mean():.4f}')
else:
    print('skip blend (missing subm_cnn.csv or submission.csv) - submission stays pure EoS.9')
'''

for code in (CNN_CELL, BLEND_CELL):
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": code.strip("\n")})

OUT = Path(__file__).parent / "birdclef-2026-eos-9-cnn.ipynb"
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", OUT, "cells:", len(nb["cells"]))
