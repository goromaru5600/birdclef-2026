# BirdCLEF 2026 Discussion 調査メモ

調査日: 2026-05-13  
現在スコア: **0.847** (XC fold0 + SC fold0 アンサンブル)  
目標: **0.88+** (銅メダル圏内)

---

## 最重要発見: PERCH v2 + ProtoSSM で 0.925 達成済み

公開ノートブック [BirdCLEF+ 2026 — Perch v2 + ProtoSSM](https://www.kaggle.com/code/imaadmahmood/birdclef-2026-perch-v2-protossm-0-925) が **0.925** を達成。  
現在の0.847はEfficientNet単体アンサンブルなので、PERCHを加えるだけで大幅向上が見込める。

別の公開ノートブック [ONNX + Perch + Proto + SED](https://www.kaggle.com/code/nina2025/birdclef-2026-onnx-perch-proto-sed) も有力。

---

## スコア向上のための施策 優先順位順

### 優先度: 高

| 手法 | 期待効果 | 難易度 |
|------|----------|--------|
| **PERCHをアンサンブルに追加** | 0.847 → 0.90+ が期待できる | 低（公開ノートブック流用） |
| **SED（Sound Event Detection）モデルのアンサンブル** | コミュニティ公開SED単体で0.86 AUC達成事例あり | 中 |
| **SC fold1/fold2を追加して4モデルアンサンブル** | 現在進行中 | 低 |

### 優先度: 中

| 手法 | 期待効果 | 難易度 |
|------|----------|--------|
| **Pseudo-labeling of soundscapes** | 未ラベルsoundscapeを擬似ラベルでXCデータに追加 | 中 |
| **BirdCLEF 2021-2024の過去データで事前学習** | 単体モデル +0.013 AUC（2025 top2%事例） | 高 |

### 優先度: 低（効果が証明されていない）

- Stratified k-fold CV（2025 top2% では効果なし）
- Focal loss（BCEより劣った）
- エネルギーベースのセグメント選択

---

## BirdCLEF 2025 top 2% (0.902 AUC) の知見

ソース: [Max Melichov Medium記事](https://medium.com/@maxme006/how-i-climbed-to-the-top-2-in-birdclef-2025-every-failure-every-lesson-and-why-details-matter-273d781a33df)

### 効いたこと

- **Quantile-Mix blending (α=0.5)** — CNNアンサンブル0.868 → 0.893 にブレイクスルー
- **Silero-VADで人声を除去** してから学習（ノイズ対策）
- **GeM pooling** (layers 3 & 4 からのmulti-layer特徴抽出)
- コミュニティ公開のSEDモデルを統合
- **5〜10epoch を超えると過学習** → 少ないエポックで十分
- 中間5秒セグメントを使用（ランダム選択よりも安定）
- 過去年度データで事前学習 → fine-tuning: 0.855 → 0.868

### 効かなかったこと（やらなくて良い）

- Wav2Vec + GNN (0.6 AUC 止まり)
- 2.5D CNNやマルチチャンネルmel (0.515 AUC)
- スクラッチからのSED構築 (0.841止まり)
- Focal loss (BCEより劣る)
- 全音声を使う (0.76 AUC 止まり)
- Stratified k-fold（効果なし）

---

## Pseudo-labeling の使い方 (Strategy Playbookより)

- 高信頼度の予測のみ採用（信頼度フィルター）
- レアクラスは手動レビュー推奨
- クリーンラベルとの比率を固定（例: 70% clean + 30% pseudo）
- Public LBではなくOOF(Out-of-Fold)でteacherモデルを選ぶと安全

---

## PERCH v2 推論時間

- TFLite変換後: **約16分** でテストサウンドスケープ全推論完了
- CPU 90分制限内でEfficientNetと組み合わせても余裕あり

---

## 次のアクション

1. SC fold1/fold2 の学習完了（RunPodで進行中）
2. SC fold1/fold2 を Kaggle Dataset にアップロード
3. 推論ノートブックを4モデルアンサンブルに更新 → LB確認
4. **PERCHノートブックを推論に追加**（最大の改善見込み）
5. 余裕があれば Pseudo-labeling を試す

---

## 参考リンク

- [BirdCLEF+ 2026 Competition](https://www.kaggle.com/competitions/birdclef-2026)
- [Perch v2 + ProtoSSM 0.925ノートブック](https://www.kaggle.com/code/imaadmahmood/birdclef-2026-perch-v2-protossm-0-925)
- [ONNX + Perch + Proto + SED ノートブック](https://www.kaggle.com/code/nina2025/birdclef-2026-onnx-perch-proto-sed)
- [BirdCLEF 2025 top2% 解法記事](https://medium.com/@maxme006/how-i-climbed-to-the-top-2-in-birdclef-2025-every-failure-every-lesson-and-why-details-matter-273d781a33df)
- [Strategy Playbook PDF](https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/Birdclef_2026.pdf)
