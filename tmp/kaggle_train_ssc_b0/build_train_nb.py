#!/usr/bin/env python3
"""Build a Kaggle GPU training notebook: distill EoS.9 pseudo-labels into an
EfficientNet-B0 trained on the SOUNDSCAPE domain.

Train  = 999 unlabeled train_soundscapes files with EoS soft labels
         (pseudo_labels.csv from kernel birdclef2026-pseudo-labels-generator).
Val    = 66 real-labeled train_soundscapes files (train_soundscapes_labels.csv)
         -> honest soundscape macro-AUC, and the exact 190-row set oof_scorer uses.

Goal: a CNN-on-mels that is soundscape-strong (unlike focal EffV2M @0.697) AND
decorrelated from Perch (the EffV2M probe showed CNNs sit at rank-corr ~0.31),
so a top-level rank blend with EoS.9 can clear the 0.950 pile.

Outputs to /kaggle/working: ssc_b0.onnx, val_preds.csv (preds on the 66 labeled
files, for the offline blend test), train_log.json
"""
import json
from pathlib import Path

CODE = r'''
import os, json, glob, math, time, random
import numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm, librosa
from sklearn.metrics import roc_auc_score

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEV, torch.__version__, 'timm', timm.__version__)

SR=32000; NFFT=1024; HOP=320; NMELS=128; FMIN=20; FMAX=16000; TFR=501; CLIP=5*SR
BACKBONE='efficientnet_b0'; EPOCHS=22; BS=64; LR=1e-3; WD=1e-4

COMP=Path('/kaggle/input/birdclef-2026')
if not COMP.exists(): COMP=Path('/kaggle/input/competitions/birdclef-2026')
SSC=COMP/'train_soundscapes'
sample=pd.read_csv(COMP/'sample_submission.csv'); LABELS=sample.columns[1:].tolist()
NC=len(LABELS); lab2i={c:i for i,c in enumerate(LABELS)}
print('classes', NC, 'soundscape files', len(list(SSC.glob('*.ogg'))))

# ---- pseudo labels (teacher) from generator kernel output ----
pl_path=glob.glob('/kaggle/input/**/pseudo_labels.csv', recursive=True)[0]
pl=pd.read_csv(pl_path)
print('pseudo_labels', pl.shape, 'from', pl_path)
pl_cols=[c for c in pl.columns if c!='row_id']
# reorder soft matrix to LABELS
assert set(LABELS).issubset(set(pl_cols)), 'pseudo cols mismatch'
SOFT = pl[LABELS].to_numpy(np.float32)
def parse_rid(r):
    st,e=r.rsplit('_',1); return st, int(e)
pl_file=[parse_rid(r)[0] for r in pl['row_id']]
pl_end =[parse_rid(r)[1] for r in pl['row_id']]

# ---- real labels (val) ----
lab=pd.read_csv(COMP/'train_soundscapes_labels.csv')
def end_sec(t):
    h,m,s=t.split(':'); return int(h)*3600+int(m)*60+int(s)
lab['end']=lab['end'].map(end_sec)
lab['file']=lab['filename'].str.replace('.ogg','',regex=False)
val_files=sorted(lab['file'].unique())
print('real-labeled val files', len(val_files))
# build val multihot per (file,end)
valY={}
for _,r in lab.iterrows():
    v=valY.setdefault((r['file'],r['end']), np.zeros(NC,np.float32))
    for t in str(r['primary_label']).split(';'):
        t=t.strip()
        if t in lab2i: v[lab2i[t]]=1.0

# ---- mel ----
def to_mel(x):
    m=librosa.feature.melspectrogram(y=x,sr=SR,n_fft=NFFT,hop_length=HOP,n_mels=NMELS,fmin=FMIN,fmax=FMAX,power=2.0)
    m=librosa.power_to_db(m,ref=np.max).astype(np.float32)
    m=(m-m.mean())/(m.std()+1e-8)
    if m.shape[1]>=TFR: m=m[:,:TFR]
    else: m=np.pad(m,((0,0),(0,TFR-m.shape[1])))
    return m

# ---- precompute mels (cache audio per file) ----
def load_file(stem):
    p=SSC/(stem+'.ogg')
    if not p.exists(): return None
    y,_=librosa.load(str(p),sr=SR); return y
def seg_mel(y,end):
    s=y[(end-5)*SR:end*SR]
    if len(s)<SR: return None
    if len(s)<CLIP: s=np.pad(s,(0,CLIP-len(s)))
    return to_mel(s)

print('precomputing TRAIN mels...'); t0=time.time()
Xtr=[]; Ytr=[]
cur=None; cache=None
order=np.argsort(pl_file)  # group by file to load each once
for k in order:
    f=pl_file[k]
    if f!=cur:
        cache=load_file(f); cur=f
    if cache is None: continue
    m=seg_mel(cache, pl_end[k])
    if m is None: continue
    Xtr.append(m.astype(np.float16)); Ytr.append(SOFT[k])
Xtr=np.stack(Xtr); Ytr=np.stack(Ytr).astype(np.float32)
print('TRAIN', Xtr.shape, 'in', round(time.time()-t0), 's')

print('precomputing VAL mels...')
Xva=[]; Yva=[]; va_rid=[]
for f in val_files:
    y=load_file(f)
    if y is None: continue
    for end in range(5,65,5):
        if (f,end) not in valY: continue
        m=seg_mel(y,end)
        if m is None: continue
        Xva.append(m.astype(np.float16)); Yva.append(valY[(f,end)]); va_rid.append(f'{f}_{end}')
Xva=np.stack(Xva); Yva=np.stack(Yva).astype(np.float32)
print('VAL', Xva.shape)

# ---- dataset ----
class DS(Dataset):
    def __init__(self,X,Y,train=False): self.X=X; self.Y=Y; self.train=train
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        x=self.X[i].astype(np.float32); y=self.Y[i]
        if self.train:
            if random.random()<0.5:  # time mask
                t=random.randint(0,40); s=random.randint(0,TFR-t); x[:,s:s+t]=0
            if random.random()<0.5:  # freq mask
                fb=random.randint(0,16); s=random.randint(0,NMELS-fb); x[s:s+fb,:]=0
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y)

tr_loader=DataLoader(DS(Xtr,Ytr,True),batch_size=BS,shuffle=True,num_workers=2,drop_last=True,pin_memory=True)
va_loader=DataLoader(DS(Xva,Yva,False),batch_size=128,shuffle=False,num_workers=2)

model=timm.create_model(BACKBONE,pretrained=True,in_chans=1,num_classes=NC).to(DEV)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS*len(tr_loader))
scaler=torch.cuda.amp.GradScaler()

def focal_bce(logits,targets,gamma=2.0):
    bce=F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    pt=torch.exp(-bce)
    return ((1-pt)**gamma*bce).mean()

def macro_auc(P,Y):
    aucs=[]
    for c in range(Y.shape[1]):
        yc=Y[:,c]
        if yc.min()==yc.max(): continue
        aucs.append(roc_auc_score(yc,P[:,c]))
    return float(np.mean(aucs)), len(aucs)

def evaluate():
    model.eval(); ps=[]
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x,_ in va_loader:
            ps.append(torch.sigmoid(model(x.to(DEV))).float().cpu().numpy())
    P=np.concatenate(ps); return P

best=0; best_P=None; log=[]
for ep in range(EPOCHS):
    model.train(); tl=0
    for x,y in tr_loader:
        x,y=x.to(DEV),y.to(DEV)
        # mixup
        if random.random()<0.5:
            lam=np.random.beta(0.4,0.4); idx=torch.randperm(x.size(0),device=DEV)
            x=lam*x+(1-lam)*x[idx]; y=lam*y+(1-lam)*y[idx]
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            loss=focal_bce(model(x),y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl+=loss.item()
    P=evaluate(); auc,nc=macro_auc(P,Yva)
    log.append({'ep':ep,'loss':tl/len(tr_loader),'val_auc':auc,'nclasses':nc})
    print(f'ep{ep:02d} loss={tl/len(tr_loader):.4f} val_macroAUC={auc:.5f} ({nc} cls)')
    if auc>best: best=auc; best_P=P; torch.save(model.state_dict(),'/kaggle/working/ssc_b0_best.pt')
print('BEST val macro-AUC', round(best,5))

# ---- save val preds (for offline blend test) + ONNX ----
vp=pd.DataFrame(best_P,columns=LABELS); vp.insert(0,'row_id',va_rid)
vp.to_csv('/kaggle/working/val_preds.csv',index=False)
json.dump(log,open('/kaggle/working/train_log.json','w'))

model.load_state_dict(torch.load('/kaggle/working/ssc_b0_best.pt'))
model.eval()
dummy=torch.randn(1,1,NMELS,TFR,device=DEV)
torch.onnx.export(model,dummy,'/kaggle/working/ssc_b0.onnx',input_names=['mel'],output_names=['logits'],
                  dynamic_axes={'mel':{0:'n'},'logits':{0:'n'}},opset_version=17)
print('WROTE ssc_b0.onnx, val_preds.csv (best AUC %.5f)'%best)
'''

# Kaggle's default torch (2.10, cu128) dropped sm_60 support; the assigned GPU is
# a P100 (sm_60). Install a torch that supports BOTH P100(sm_60) and T4(sm_75)
# BEFORE importing torch. Must be its own cell so the later import picks it up.
INSTALL = (
    "import subprocess, sys\n"
    "subprocess.run([sys.executable,'-m','pip','install','-q',\n"
    "  'torch==2.4.1','torchvision==0.19.1',\n"
    "  '--index-url','https://download.pytorch.org/whl/cu121'], check=True)\n"
    "import torch; print('reinstalled torch', torch.__version__, 'arch', torch.cuda.get_arch_list())\n"
    "assert torch.cuda.is_available()\n"
    "a=torch.randn(8,8,device='cuda'); print('CUDA OK', float((a@a).sum()))"
)

nb = {
    "cells": [
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": INSTALL},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CODE.strip("\n")},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(__file__).parent / "train-ssc-b0.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", out)
