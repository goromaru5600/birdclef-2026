# BirdCLEF+ 2026 銅メダル獲得戦略

## Context

BirdCLEF+ 2026はパンタナル湿地（南米ブラジル）の音響種識別コンペ。**残り期間約3.5週間**（エントリー締切: 5/27、最終提出: 6/3）。現在340チーム参加中で、**銅メダル圏内 = 上位約34チーム（上位10%）** が目標。

**評価指標**: Macro-averaged ROC-AUC（234クラス、真陽性なしのクラスはスキップ）  
**制約**: CPU onlyノートブック、90分以内の推論  
**出力形式**: 5秒ウィンドウごとの234種の予測確率

---

## 競合状況と銅メダルの目安スコア

- 参加チーム数: ~340チーム → 銅メダルライン ≈ **34位以内**
- BirdCLEF 2025参考: 銅メダル圏内のROC-AUCは約0.89〜0.90台
- PERCH v2ベースライン単体: ~0.73、BirdSetEfficientNetB1: ~0.81
- 目標スコア: **0.88〜0.91+**（アンサンブル）

---

## 実装計画（3フェーズ）

### Phase 1: データ準備とベースライン（Week 1: 5/11〜5/17）

#### ステップ1-1: データダウンロード
```bash
kaggle competitions download -c birdclef-2026 -p ./data
```
ファイル: `./data/`（train_audio/, test_soundscapes/, train_metadata.csv）

#### ステップ1-2: EDAノートブック作成
- `notebooks/eda.ipynb`: クラス分布、音声長、ラベル共起分析
- 234クラスの分布確認（ロングテール問題の把握）

#### ステップ1-3: PERCH v2ベースラインの提出
- `notebooks/baseline_perch.ipynb` を作成（公開ノートブック `kailyn2359/birdclef-2026-baseline-with-perch-v2-model` を参考）
- **目標: 0.73以上でスコア確認**

---

### Phase 2: カスタムモデル学習（Week 1〜2: 5/13〜5/25）

#### ステップ2-1: メルスペクトログラムパイプライン
ファイル: `models/preprocess.py`
```python
# 主要パラメータ
SAMPLE_RATE = 32000
N_FFT = 1024
HOP_LENGTH = 320      # → 100fps
N_MELS = 128
DURATION = 5          # 秒（1チャンク）
```

#### ステップ2-2: EfficientNet-B0/B1 ファインチューニング
ファイル: `models/train.py`
- **ベースモデル**: `timm` の `efficientnet_b0` or `efficientnet_b1`
- **入力**: 128×157 メルスペクトログラム（5秒）
- **出力**: 234クラス sigmoid
- **損失関数**: BCEWithLogitsLoss or FocalLoss（クラス不均衡対策）
- **最適化**: AdamW + CosineAnnealingLR with warmup
- **CV**: 5-fold Stratified（種ラベルベース）

**Augmentation（学習時）**:
- SpecAugment（時間・周波数マスク）
- Mixup (alpha=0.4)
- ランダムクロップ（音声長変動対応）
- ランダムノイズ付加

#### ステップ2-3: ONNX変換（CPU推論最適化）
ファイル: `models/export_onnx.py`
- EfficientNetをONNXに変換してCPU推論を高速化
- 目標: 5秒音声1チャンクの推論 < 50ms

---

### Phase 3: アンサンブルと提出最適化（Week 3: 5/25〜6/3）

#### ステップ3-1: アンサンブル設計
ファイル: `notebooks/inference_ensemble.ipynb`（Kaggleで提出するノートブック）

```
アンサンブル構成（90分以内に収める）:
1. PERCH v2 (TFLite) ── 推論時間 ~16分
2. EfficientNet-B0 × 3fold (ONNX) ── 推論時間 ~30分
3. EfficientNet-B1 × 2fold (ONNX) ── 推論時間 ~30分
合計 ~76分（マージン確保）

アンサンブル方法: 各モデル出力の重み付き平均
```

#### ステップ3-2: 後処理
- **しきい値なし提出**（ROC-AUCはスコア連続値で評価）
- テストサウンドスケープを5秒チャンクに分割 → 各チャンクで全種予測 → row_id ごとに最大値or平均で集約

#### ステップ3-3: 最終提出ノートブック
ファイル: `submissions/final_inference.ipynb`
- CPU only、90分制限を必ず確認してからサブミット

---

## 重要ファイル一覧

| ファイル | 目的 |
|---------|------|
| `data/` | コンペデータ（DL済み後） |
| `notebooks/eda.ipynb` | EDA |
| `notebooks/baseline_perch.ipynb` | PERCH v2ベースライン |
| `models/preprocess.py` | 音声→メルスペクトログラム変換 |
| `models/train.py` | EfficientNet学習スクリプト |
| `models/export_onnx.py` | ONNX変換 |
| `submissions/final_inference.ipynb` | 最終提出ノートブック |

---

## 参考リソース（再利用推奨）

- PERCH v2 starter: `kaggle.com/code/zirach/birdclef-2026-perch-v2-starter-readable-ver`
- PERCH v2 baseline: `kaggle.com/code/kailyn2359/birdclef-2026-baseline-with-perch-v2-model`
- Robust submission starter: `kaggle.com/code/dedquoc/birdclef-2026-the-robust-submission-starter`
- 前年銅メダル実装: `github.com/AswinKumar1/Kaggle-Bronze-winner-Bird-CLEF-2025`

---

## 検証方法

1. **ローカルCV**: 5-fold の OOF ROC-AUCで学習モニタリング
2. **Kaggle Public LB**: 各フェーズ終了時にサブミットしてスコア確認
3. **推論時間確認**: Kaggle CPU notebookで実際に90分以内に完了することを確認してからサブミット
4. **Private LB**: 最終締切後に確定（Public LBとの乖離に注意）

---

## タイムライン

| 日程 | マイルストーン |
|------|--------------|
| 5/11〜5/12 | データDL・EDA・PERCH v2ベースライン提出 |
| 5/13〜5/18 | EfficientNet学習（B0、5fold） |
| 5/19〜5/22 | アンサンブルノートブック作成・スコア確認 |
| 5/23〜5/25 | EfficientNet-B1追加学習・チューニング |
| 5/26 | エントリー締切（チームマージ最終日） |
| 5/27〜6/2 | 最終アンサンブル調整・サブミット繰り返し |
| 6/3 | **最終サブミット締切** |
