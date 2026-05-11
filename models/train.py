"""
BirdCLEF+ 2026 — EfficientNet Training Script
Run this on Kaggle GPU notebook.

Usage on Kaggle:
    !python train.py --fold 0 --epochs 20 --model efficientnet_b0
"""
import os, sys, time, warnings, argparse
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
COMP_DIR  = os.environ.get('COMP_DIR', '/kaggle/input/birdclef-2026')
OUT_DIR   = os.environ.get('OUT_DIR', '/kaggle/working')

SR        = 32000
DURATION  = 5        # seconds per chunk
N_FFT     = 1024
HOP_LEN   = 320      # 32000/320 = 100 fps
N_MELS    = 128
FMIN      = 20
FMAX      = 16000

IMG_H     = 128
IMG_W     = 160      # ~5s at 100fps → 500 frames → crop/resize to 160
BATCH     = 32
LR        = 3e-4
NUM_WORKERS = 2

# ── Parse args ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fold',   type=int,   default=0)
    p.add_argument('--epochs', type=int,   default=20)
    p.add_argument('--model',  type=str,   default='efficientnet_b0')
    p.add_argument('--seed',   type=int,   default=42)
    return p.parse_args()

# ── Imports ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import albumentations as A
from torch.cuda.amp import autocast, GradScaler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ── Audio → Mel Spectrogram ──────────────────────────────────────────────
def audio_to_melspec(audio, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_norm.astype(np.float32)

def load_random_chunk(path, duration=DURATION, sr=SR):
    """Load a random 5-second chunk from an audio file."""
    total_samples = int(librosa.get_duration(path=path) * sr)
    chunk_samples = duration * sr
    if total_samples <= chunk_samples:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        audio = np.pad(audio, (0, max(0, chunk_samples - len(audio))))
    else:
        max_start = total_samples - chunk_samples
        start = np.random.randint(0, max_start)
        audio, _ = librosa.load(path, sr=sr, mono=True,
                                offset=start/sr, duration=duration)
    return audio[:chunk_samples]

# ── SpecAugment ──────────────────────────────────────────────────────────
def spec_augment(mel, n_freq_masks=2, n_time_masks=2, freq_max=15, time_max=25):
    mel = mel.copy()
    _, T = mel.shape
    for _ in range(n_freq_masks):
        f = np.random.randint(0, freq_max)
        f0 = np.random.randint(0, N_MELS - f)
        mel[f0:f0+f, :] = 0.0
    for _ in range(n_time_masks):
        t = np.random.randint(0, time_max)
        t0 = np.random.randint(0, max(1, T - t))
        mel[:, t0:t0+t] = 0.0
    return mel

# ── Dataset ──────────────────────────────────────────────────────────────
class BirdDataset(Dataset):
    def __init__(self, df, species_list, augment=False):
        self.df = df.reset_index(drop=True)
        self.species_list = species_list
        self.label_to_idx = {s: i for i, s in enumerate(species_list)}
        self.augment = augment
        self.n_classes = len(species_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(COMP_DIR, 'train_audio', row['filename'])

        try:
            audio = load_random_chunk(path)
        except Exception:
            audio = np.zeros(DURATION * SR, dtype=np.float32)

        mel = audio_to_melspec(audio)

        # Resize to (IMG_H, IMG_W)
        if mel.shape[1] != IMG_W:
            mel = np.array([np.interp(
                np.linspace(0, mel.shape[1]-1, IMG_W),
                np.arange(mel.shape[1]), mel[i]
            ) for i in range(mel.shape[0])], dtype=np.float32)

        if self.augment:
            mel = spec_augment(mel)

        # Repeat to 3 channels for ImageNet-pretrained backbone
        img = np.stack([mel, mel, mel], axis=0)  # (3, H, W)

        # Multi-label target
        label = np.zeros(self.n_classes, dtype=np.float32)
        primary = str(row['primary_label'])
        if primary in self.label_to_idx:
            label[self.label_to_idx[primary]] = 1.0

        # Secondary labels
        sec = row.get('secondary_labels', '[]')
        if isinstance(sec, str) and sec not in ('[]', ''):
            import ast
            try:
                for s in ast.literal_eval(sec):
                    s = str(s)
                    if s in self.label_to_idx:
                        label[self.label_to_idx[s]] = 0.5  # soft label
            except Exception:
                pass

        return torch.from_numpy(img), torch.from_numpy(label)

# ── Model ────────────────────────────────────────────────────────────────
class BirdModel(nn.Module):
    def __init__(self, model_name, n_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=3,
            num_classes=0, global_pool='avg'
        )
        n_feat = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_feat, n_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

# ── Mixup ────────────────────────────────────────────────────────────────
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y + (1 - lam) * y[idx]
    return mixed_x, mixed_y

# ── Training loop ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    losses = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs, labels = mixup(imgs, labels, alpha=0.4)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    return np.mean(losses)

@torch.no_grad()
def valid_epoch(model, loader, criterion, species_list):
    model.eval()
    all_preds, all_labels, losses = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        losses.append(loss.item())

    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    # Macro ROC-AUC (skip classes with no positives)
    aucs = []
    for i in range(labels.shape[1]):
        if labels[:, i].sum() > 0:
            aucs.append(roc_auc_score(labels[:, i], preds[:, i]))
    macro_auc = np.mean(aucs) if aucs else 0.0
    return np.mean(losses), macro_auc

# ── Main ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_df = pd.read_csv(os.path.join(COMP_DIR, 'train.csv'))
    taxonomy = pd.read_csv(os.path.join(COMP_DIR, 'taxonomy.csv'))
    sub_df   = pd.read_csv(os.path.join(COMP_DIR, 'sample_submission.csv'))

    # Full species list from submission columns
    species_list = [c for c in sub_df.columns if c != 'row_id']
    print(f'Total species: {len(species_list)}')
    print(f'Train records: {len(train_df)}')

    # Filter low-quality recordings (optional: keep rating >= 3 or all)
    # train_df = train_df[train_df['rating'] >= 3].reset_index(drop=True)

    # 5-fold CV — stratify on primary_label
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    splits = list(skf.split(train_df, train_df['primary_label']))
    train_idx, val_idx = splits[args.fold]

    train_fold = train_df.iloc[train_idx]
    val_fold   = train_df.iloc[val_idx]
    print(f'Fold {args.fold}: train={len(train_fold)}, val={len(val_fold)}')

    train_ds = BirdDataset(train_fold, species_list, augment=True)
    val_ds   = BirdDataset(val_fold,   species_list, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = BirdModel(args.model, n_classes=len(species_list)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=LR * 0.01
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_auc = 0.0
    save_path = os.path.join(OUT_DIR, f'model_fold{args.fold}.pt')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, criterion)
        val_loss, val_auc = valid_epoch(model, val_loader, criterion, species_list)
        scheduler.step()

        elapsed = time.time() - t0
        print(f'Epoch {epoch:02d} | '
              f'train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | '
              f'val_auc={val_auc:.4f} | {elapsed:.0f}s')

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_auc': val_auc,
                'species_list': species_list,
                'model_name': args.model,
            }, save_path)
            print(f'  -> Saved best model (auc={val_auc:.4f})')

    print(f'\nBest val AUC: {best_auc:.4f}')
    print(f'Model saved to: {save_path}')

if __name__ == '__main__':
    main()
