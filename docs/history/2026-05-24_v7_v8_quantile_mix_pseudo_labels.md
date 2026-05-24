# 2026-05-24 作業履歴: v7/v8 push + 疑似ラベル生成パイプライン

## 開始時の状況

- **目標**: LB 0.955 (銅メダル)
- **現状**: v5=0.949, v6=0.948 (B1追加で悪化), 残り日数 ~10日 (締切 2026-06-03)
- **アンサンブル構成**: ProtoSSM 60% + Tucker B0 SED 40% (rank-blend)

## 実施内容

### 1. v7 push: B1 SED完全除去 (v5相当に復帰)

**動機**: v6でB1 SED fold 1+2 (10% weight) 追加 → LB 0.948 と悪化。原因はB0/B1の高相関＋128mel/256melの前処理差と推定。一旦0.949基準を確保。

**変更**:
- `tmp/kaggle_push_eos5/birdclef-2026-eos-5.ipynb` cell 13:
  - B1 SED loading (audio_to_mel_128, sed_sessions_b1, B1_WEIGHT) 削除
  - 推論ループの B1 prediction block 削除 (`p_mean = p_b0` のみに)
- `tmp/kaggle_push_eos5/kernel-metadata.json`:
  - `gorubachohu/560-sed-distill-fold0` を dataset_sources から削除

**結果**: v7 push成功 (kernelId=119788727, versionNumber=7)
- URL: https://www.kaggle.com/code/gorubachohu/birdclef2026-eos5-fork

**学び (Kaggle API)**:
- KGAT_ tokenでは `slug` フィールドが必要 (full `"user/slug"` 形式)
- 最初 `slug="birdclef2026-eos5-fork"` → "Invalid slug" エラー
- 正解: `slug="gorubachohu/birdclef2026-eos5-fork"`
- Notebookセル: `outputs` を空配列に、`source` リストを文字列に正規化必須

### 2. B案: アンサンブル重みグリッド探索

**手法**: dry-run出力 (`submission_protossm.csv`, `submission_sed.csv`) と `train_soundscapes_labels.csv` を突合
- 評価データ: 240 chunks → ラベル重なりあり 190 chunks × 42 classes (macro-AUC)

**結果**:

| w_proto | w_sed | Rank-mean AUC |
|---------|-------|---------------|
| 0.10 | 0.90 | **0.99612** (best) |
| 0.50 | 0.50 | 0.99436 |
| **0.60** | **0.40** | **0.99246** (現在) |

**判断**: データリーク懸念 (公開SEDモデルがtrain_soundscapesで学習されている可能性高) のため OOFを信頼せず、weight変更は見送り

### 3. C案: per-class threshold tuning

**調査結果**: ノートブックに**既に実装済み**
- `calibrate_and_optimize_thresholds()` (cell 13 L923) — F1最適化 + isotonic校正
- 細粒度grid: `[0.20-0.45 step 0.025] + [0.45-0.75 step 0.05]`
- `apply_per_class_thresholds()` 線形ピースワイズ sharpening
- rare-class adaptive thresholding (Gate 5)

**判断**: 余地小、スキップ

### 4. A案: 公開Notebook調査 (subagent並行実行)

**主要発見 (推奨度順)**:

| # | 手法 | 期待値 | 難易度 |
|---|------|--------|--------|
| 1 | Multi-year Xeno-Canto pretrain (2021-2024) | +0.013〜0.026 | 中 |
| 2 | **Quantile-Mix blend** (post-process) | **+0.025** | **低** |
| 3 | **Soundscape pseudo-labeling (2 iter)** | **+0.015〜0.020** | **中** |
| 4 | Center-5s + multi-scale mel diversity | +0.013〜0.020 | 中 |
| 5 | Silero-VAD 人声除去 | 非公開 | 低 |
| 6 | rare-class head re-training | 不明 | 中 |
| 7 | Site/Hour priors | TBD | 低 |

**出典**:
- BirdCLEF 2025 2nd place: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
- Top 2% deep-dive: https://medium.com/@maxme006/how-i-climbed-to-the-top-2-in-birdclef-2025-...
- ferariz 2026 ablation: https://github.com/ferariz/birdclef2026
- 2nd place paper: https://ceur-ws.org/Vol-4038/paper_256.pdf

### 5. D案 Phase 1: 疑似ラベル生成 Notebook

**設計**: EoS5を流用してtrain_soundscapes全件 (~150 files) で推論

**変更** (`tmp/kaggle_push_pseudo/pseudo-labels.ipynb`):
- Cell 7 L120: `dryrun_n_files: 20 → 9999` (全train_soundscapes処理)
- Cell 13 L172: 同上
- Cell 13 rank-blend直後にCSV save:
  ```python
  pseudo_prob = (p_proto * PROTO_W + p_sed * SED_W)  # 確率空間ブレンド
  pseudo_df.to_csv("/kaggle/working/pseudo_labels.csv")
  ```

**結果**: push成功 (kernelId=120370704)
- URL: https://www.kaggle.com/code/gorubachohu/birdclef2026-pseudo-labels-generator
- 推定実行時間: 10-15分 (Kaggle CPU)
- 期待出力: ~1800 chunks × 234 classes

### 6. v8 push: Quantile-Mix blend 実装 (A案 #2)

**動機**: rank-meanを正規分布写像ベースのblendに置換

**OOF比較** (190 chunks × 42 classes):

| Method | w=0.30 | w=0.50 | w=0.60 (現在) |
|--------|--------|--------|---------------|
| Rank-mean (v7) | 0.99552 | 0.99431 | **0.99274** |
| Quantile-Mix (v8) | **0.99601** | 0.99493 | **0.99382** |
| 差分 | +0.0005 | +0.0006 | **+0.0011** |

**実装** (cell 13 rank-blend部置換):
```python
# 旧: pred = (rank_proto * PROTO_W) + (rank_sed * SED_W)
# 新:
from scipy.stats import norm
EPS_Q = 1e-6
q_proto = norm.ppf(np.clip(rank_proto, EPS_Q, 1.0 - EPS_Q))
q_sed   = norm.ppf(np.clip(rank_sed,   EPS_Q, 1.0 - EPS_Q))
q_blend = PROTO_W * q_proto + SED_W * q_sed
pred = norm.cdf(q_blend)
```

**結果**: v8 push成功 (versionNumber=8)
- 期待LB: 0.949 → 0.950〜0.951

## 成果サマリー

| 項目 | 状態 |
|------|------|
| v7 (B1除去) | Pushed, 未Submit |
| v8 (Quantile-Mix) | Pushed, 未Submit |
| 疑似ラベル生成 notebook | Pushed, **未実行** |
| Phase 2 (B1再学習) | 設計済、Phase 1出力待ち |
| Phase 3 (統合v9 push) | 未着手 |

## 次のアクション (ユーザー実行)

1. **Kaggleで v8 をSubmit** (主要、+0.001〜0.002期待)
2. **疑似ラベル生成Notebookを実行** (10-15分)
3. v7 もSubmit可 (基準値確認用)

## 次のアクション (アシスタント実行予定)

1. **Phase 2 設計詳細化**: B1学習notebook (`bc2026-sed-colab.ipynb`) の改変
   - S2 cell: pseudo_labels.csv ロード追加
   - Y_SC を hybrid 化 (実ラベル優先 + 疑似ラベル補完)
   - fold 0/3/4 再学習 (OOF低fold)
2. v8 LB結果次第で他施策検討
   - A案 #5 (Silero-VAD): +微増可能性
   - A案 #7 (Site/Hour priors): +微増可能性

## 参考ファイル

- 修正済 notebook: `tmp/kaggle_push_eos5/birdclef-2026-eos-5.ipynb`
- 疑似ラベル生成: `tmp/kaggle_push_pseudo/pseudo-labels.ipynb`
- B1学習 (改変対象): `tmp/colab_tucker_sed_b1/bc2026-sed-colab.ipynb`
- OOF評価データ: `tmp/kaggle_push_eos5/eval_data.npz` (P, S, Y, class_mask)
- Push スクリプト: `tmp/kaggle_push_eos5/push_v7.py`, `push_v8.py`
- Kaggle API auth: KGAT_ token + Bearer auth (`/kaggle/api/v1/kernels/push`)
