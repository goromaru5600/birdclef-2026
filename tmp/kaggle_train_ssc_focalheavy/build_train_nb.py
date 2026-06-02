#!/usr/bin/env python3
"""v2 training: 2-stage to make the B0 STRONGER and MORE DECORRELATED from EoS.9.

Stage A (focal pretrain): train_audio (real multi-hot, primary+secondary), balanced
  subset capped K/class -> independent signal covering 206/234 classes.
Stage B (soundscape fine-tune): 999-file EoS pseudo-labels (soft) -> domain adapt.
Val: 66 real-labeled soundscape files (same 190-row harness as v3, comparable).

v3 (pure distillation) gave soundscape val AUC 0.872, rank-corr(EoS.9)=0.551,
blend +0.0001@w0.05. Focal real labels should raise strength AND lower correlation.

Outputs: ssc_focalheavy.onnx, val_preds.csv, train_log.json
"""
import json
from pathlib import Path

INSTALL = (
    "import subprocess, sys\n"
    "subprocess.run([sys.executable,'-m','pip','install','-q',\n"
    "  'torch==2.4.1','torchvision==0.19.1',\n"
    "  '--index-url','https://download.pytorch.org/whl/cu121'], check=True)\n"
    "import torch; print('torch', torch.__version__, torch.cuda.get_arch_list())\n"
    "assert torch.cuda.is_available(); print('CUDA OK', float((torch.randn(8,8,device='cuda')**2).sum()))"
)

CODE = r'''
import os, json, glob, ast, time, random
import numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm, librosa
from sklearn.metrics import roc_auc_score

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEV='cuda'
SR=32000; NFFT=1024; HOP=320; NMELS=128; FMIN=20; FMAX=16000; TFR=501; CLIP=5*SR
BACKBONE='efficientnet_b0'; BS=64
K_PER_CLASS=80                 # focal cap per class
EP_A=12; LR_A=1e-3             # Stage A focal pretrain
EP_B=5; LR_B=4e-4             # Stage B soundscape fine-tune

COMP=Path('/kaggle/input/birdclef-2026')
if not COMP.exists(): COMP=Path('/kaggle/input/competitions/birdclef-2026')
TRAUD=COMP/'train_audio'; SSC=COMP/'train_soundscapes'
sample=pd.read_csv(COMP/'sample_submission.csv'); LABELS=sample.columns[1:].tolist()
NC=len(LABELS); lab2i={c:i for i,c in enumerate(LABELS)}
print('classes', NC)

def to_mel(x):
    m=librosa.feature.melspectrogram(y=x,sr=SR,n_fft=NFFT,hop_length=HOP,n_mels=NMELS,fmin=FMIN,fmax=FMAX,power=2.0)
    m=librosa.power_to_db(m,ref=np.max).astype(np.float32)
    m=(m-m.mean())/(m.std()+1e-8)
    if m.shape[1]>=TFR: m=m[:,:TFR]
    else: m=np.pad(m,((0,0),(0,TFR-m.shape[1])))
    return m

# ---------- Stage A data: focal ----------
tr=pd.read_csv(COMP/'train.csv')
tr=tr.groupby('primary_label',group_keys=False).apply(lambda d: d.sample(min(len(d),K_PER_CLASS),random_state=SEED))
print('focal subset', len(tr))
def focal_multihot(row):
    v=np.zeros(NC,np.float32)
    if row['primary_label'] in lab2i: v[lab2i[row['primary_label']]]=1.0
    try:
        for s in ast.literal_eval(row['secondary_labels']) or []:
            if s in lab2i: v[lab2i[s]]=1.0
    except Exception: pass
    return v
print('precompute focal mels...'); t0=time.time()
XA=[]; YA=[]
for _,row in tr.iterrows():
    p=TRAUD/row['filename']
    if not p.exists(): continue
    try: y,_=librosa.load(str(p),sr=SR,duration=10.0)
    except Exception: continue
    if len(y)<SR: continue
    c=len(y)//2; s=max(0,c-CLIP//2); seg=y[s:s+CLIP]
    if len(seg)<CLIP: seg=np.pad(seg,(0,CLIP-len(seg)))
    XA.append(to_mel(seg).astype(np.float16)); YA.append(focal_multihot(row))
XA=np.stack(XA); YA=np.stack(YA).astype(np.float32)
print('focal', XA.shape, round(time.time()-t0),'s')

# ---------- Stage B data: soundscape pseudo ----------
pl=pd.read_csv(glob.glob('/kaggle/input/**/pseudo_labels.csv',recursive=True)[0])
SOFT=pl[LABELS].to_numpy(np.float32)
pf=[r.rsplit('_',1)[0] for r in pl['row_id']]; pe=[int(r.rsplit('_',1)[1]) for r in pl['row_id']]
def load_ssc(stem):
    p=SSC/(stem+'.ogg');
    if not p.exists(): return None
    y,_=librosa.load(str(p),sr=SR); return y
def seg_mel(y,end):
    s=y[(end-5)*SR:end*SR]
    if len(s)<SR: return None
    if len(s)<CLIP: s=np.pad(s,(0,CLIP-len(s)))
    return to_mel(s)
print('precompute soundscape pseudo mels...'); t0=time.time()
XB=[]; YB=[]; cur=None; cache=None
for k in np.argsort(pf):
    f=pf[k]
    if f!=cur: cache=load_ssc(f); cur=f
    if cache is None: continue
    m=seg_mel(cache,pe[k])
    if m is None: continue
    XB.append(m.astype(np.float16)); YB.append(SOFT[k])
XB=np.stack(XB); YB=np.stack(YB).astype(np.float32)
print('pseudo', XB.shape, round(time.time()-t0),'s')

# ---------- Val: 66 real ----------
lab=pd.read_csv(COMP/'train_soundscapes_labels.csv')
def es(t): h,m,s=t.split(':'); return int(h)*3600+int(m)*60+int(s)
lab['end']=lab['end'].map(es); lab['file']=lab['filename'].str.replace('.ogg','',regex=False)
valY={}
for _,r in lab.iterrows():
    v=valY.setdefault((r['file'],r['end']),np.zeros(NC,np.float32))
    for t in str(r['primary_label']).split(';'):
        t=t.strip()
        if t in lab2i: v[lab2i[t]]=1.0
XV=[]; YV=[]; vrid=[]
for f in sorted(lab['file'].unique()):
    y=load_ssc(f)
    if y is None: continue
    for end in range(5,65,5):
        if (f,end) not in valY: continue
        m=seg_mel(y,end)
        if m is None: continue
        XV.append(m.astype(np.float16)); YV.append(valY[(f,end)]); vrid.append(f'{f}_{end}')
XV=np.stack(XV); YV=np.stack(YV).astype(np.float32)
print('val', XV.shape)

class DS(Dataset):
    def __init__(s,X,Y,tr=False): s.X=X; s.Y=Y; s.tr=tr
    def __len__(s): return len(s.X)
    def __getitem__(s,i):
        x=s.X[i].astype(np.float32); y=s.Y[i]
        if s.tr:
            if random.random()<0.5:
                t=random.randint(0,40); a=random.randint(0,TFR-t); x[:,a:a+t]=0
            if random.random()<0.5:
                fb=random.randint(0,16); a=random.randint(0,NMELS-fb); x[a:a+fb,:]=0
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y)

va=DataLoader(DS(XV,YV),batch_size=128,num_workers=2)
model=timm.create_model(BACKBONE,pretrained=True,in_chans=1,num_classes=NC).to(DEV)
scaler=torch.cuda.amp.GradScaler()
def focal_bce(logits,t,g=2.0):
    bce=F.binary_cross_entropy_with_logits(logits,t,reduction='none'); pt=torch.exp(-bce)
    return ((1-pt)**g*bce).mean()
def macro_auc(P,Y):
    a=[roc_auc_score(Y[:,c],P[:,c]) for c in range(Y.shape[1]) if Y[:,c].min()!=Y[:,c].max()]
    return float(np.mean(a)), len(a)
def evaluate():
    model.eval(); ps=[]
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x,_ in va: ps.append(torch.sigmoid(model(x.to(DEV))).float().cpu().numpy())
    return np.concatenate(ps)

def run_stage(X,Y,epochs,lr,tag,best,bestP):
    ld=DataLoader(DS(X,Y,True),batch_size=BS,shuffle=True,num_workers=2,drop_last=True,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs*len(ld))
    for ep in range(epochs):
        model.train(); tl=0
        for x,y in ld:
            x,y=x.to(DEV),y.to(DEV)
            if random.random()<0.5:
                lam=np.random.beta(0.4,0.4); idx=torch.randperm(x.size(0),device=DEV)
                x=lam*x+(1-lam)*x[idx]; y=lam*y+(1-lam)*y[idx]
            opt.zero_grad()
            with torch.cuda.amp.autocast(): loss=focal_bce(model(x),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); tl+=loss.item()
        P=evaluate(); auc,n=macro_auc(P,YV)
        LOG.append({'stage':tag,'ep':ep,'loss':tl/len(ld),'val_auc':auc})
        print(f'[{tag}] ep{ep:02d} loss={tl/len(ld):.4f} val_AUC={auc:.5f} ({n})')
        if auc>best: best=auc; bestP=P; torch.save(model.state_dict(),'/kaggle/working/ssc_focalheavy_best.pt')
    return best,bestP

LOG=[]; best=0; bestP=None
best,bestP=run_stage(XA,YA,EP_A,LR_A,'focal',best,bestP)
best,bestP=run_stage(XB,YB,EP_B,LR_B,'ssc',best,bestP)
print('BEST val macro-AUC', round(best,5))

pd.DataFrame(bestP,columns=LABELS).assign(row_id=vrid)[['row_id']+LABELS].to_csv('/kaggle/working/val_preds.csv',index=False)
json.dump(LOG,open('/kaggle/working/train_log.json','w'))
model.load_state_dict(torch.load('/kaggle/working/ssc_focalheavy_best.pt')); model.eval()
torch.onnx.export(model,torch.randn(1,1,NMELS,TFR,device=DEV),'/kaggle/working/ssc_focalheavy.onnx',
                  input_names=['mel'],output_names=['logits'],dynamic_axes={'mel':{0:'n'},'logits':{0:'n'}},opset_version=17)
print('WROTE ssc_focalheavy.onnx, val_preds.csv  best=%.5f'%best)
'''

nb = {"cells": [
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": INSTALL},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CODE.strip("\n")},
], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "train-ssc-focalheavy.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", out)
