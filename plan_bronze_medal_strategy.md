# BirdCLEF 2026 Bronze Medal Strategy (0.949 → 0.96)

作成日: 2026-05-20

## 現状

| 項目 | 内容 |
|---|---|
| 現在の LB スコア | ~0.949 |
| 目標 LB スコア | 0.955 |
| 締め切り | 2026-06-03 |
| 推論制約 | CPU only, 90分以内 |
| Public notebook 最高 | 0.946〜0.948 |

**0.949 はすでに公開 notebook の最上位水準。0.96 超えは非公開アプローチを持つ上位陣のみ確認されており、挑戦的な目標。**

---

## 現在の構成 (EoS5 fork ベース)

```
Perch ONNX → ProtoSSM + MLP probe + ResidualSSM  → 60%
Tucker SED B0 5-fold ONNX                          → 40%
2-way rank blend
+ Sonotype mirroring (10 クラス)
+ Adaptive thresholding (44 レア種)
```

---

## タイムライン（残り 14 日）

```
05/20 (今日)  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  06/03 (締切)
  │
  ├─[Week 1: 05/20〜05/26] B1 SED 完成 + 初回 LB 確認
  │   05/20  B1 fold 0 学習中 (v14, T4 x2)
  │   05/21  fold 0 完了 → ONNX 取得 → Kaggle dataset にアップロード
  │   05/22  fold 1 学習 → ONNX アップロード
  │   05/23  fold 2 学習 → ONNX アップロード
  │   05/24  fold 3 学習 → ONNX アップロード
  │   05/25  fold 4 学習 → ONNX アップロード
  │   05/25  EoS5 fork に B1 SED 組み込み → LB 提出 (目標 0.950〜0.952)
  │
  ├─[Week 2a: 05/26〜05/29] 擬似ラベル再学習
  │   05/26  B1 アンサンブルで test soundscape を予測 → 擬似ラベル生成
  │   05/27  B1 SED をバックボーン freeze で 20 epoch 再学習 (5 fold)
  │   05/28  再学習 ONNX を EoS5 に組み込み → LB 提出 (目標 0.958〜0.960)
  │
  ├─[Week 2b: 05/29〜06/01] 純 CNN 追加（余力があれば）
  │   05/29  EfficientNetv2_s を Perch 蒸留なしで訓練 (5 fold)
  │   05/31  EoS5 に 3-way blend で追加 → LB 提出
  │   06/01  アンサンブル比率の OOF 最適化
  │
  └─[最終調整: 06/02〜06/03]
      06/02  最終 submission の選定 (best LB vs 安定スコア)
      06/03  締切 23:59
```

**各フェーズの判断基準**
- fold 0 OOF が B0 と同等以下 → B0/B1 両方をアンサンブルに入れる
- 擬似ラベル後の LB が下がる → フィルタリング閾値を調整して再試行
- 純 CNN の単体 LB が 0.920 未満 → アンサンブルから除外

---

## 改善ロードマップ

### フェーズ 1: Tucker B1 SED 完成（進行中）

**作業内容**
- `birdclef2026-tucker-sed-b1` notebook で fold 0〜4 を順次学習
- 各 fold の `sed_distill_fold{k}.onnx` を Kaggle dataset にアップロード
- EoS5 fork の Tucker SED を B0 → B1 に差し替えて LB 提出

**期待効果**: +0.001〜+0.003

**判断基準**
- fold 0 OOF スコアが B0 より高ければ B1 に移行
- 差がなければ B0/B1 の両方をアンサンブルに入れる（多様性として）

**現在の学習設定**
```python
BACKBONE_NAME = "tf_efficientnet_b1.ns_jft_in1k"
FOLDS  = [0]     # 1 fold ずつ順番に実行
EPOCHS = 20
BATCH  = 16      # T4 x2 での RAM OOM 回避
USE_PERCH_DISTILL = True
```

---

### フェーズ 2: 擬似ラベル再学習

**最も即効性が高い手法。HGNetV2-B0 で 0.931 → 0.943 (+0.012) の実績あり。**

**手順**
1. 現在のアンサンブルでテスト soundscape を予測
2. 予測結果をソフトラベルのまま訓練データに連結（フィルタリングなし）
3. バックボーン freeze のまま 20 epoch 再学習（全体 fine-tune は逆効果）
4. 再学習後の ONNX を EoS5 に組み込んで LB 提出

**重要な注意点**
- フィルタリング（confidence/entropy による絞り込み）は逆効果と複数人が報告
- ソフトラベル維持・単純連結が最善
- バックボーン freeze または極低 LR での微調整のみ

**期待効果**: +0.008〜+0.012

---

### フェーズ 3: 純 CNN モデルの追加（多様性確保）

**Perch 蒸留なし**の独立した CNN を追加してアンサンブルの多様性を確保する。

**候補モデル**
- EfficientNetv2_s（Perch 蒸留なし）
- HGNetV2-B0（ttahara 系、LSE pooling 採用）

**特徴**
- 単体で 0.936 達成の報告あり
- Perch 系と誤差パターンが異なるため相関低い
- アンサンブルに追加で +0.002〜+0.005 の可能性

**BirdNET は逆効果**（AUC 0.67、Perch 系より大幅に劣る。複数人が LB 低下を確認）

**期待効果**: +0.002〜+0.005

---

### フェーズ 4: アーキテクチャ改善（優先度低）

#### LSE (LogSumExp) Pooling の採用
- Tucker SED の GeMFreqPool → LSE Pool に変更
- ttahara の実験で LB +0.016
- ただし SED head 全体の再実装が必要でコスト高

#### 蒸留先の変更（学習高速化）
- Perch v2 の最終 1536-d head ではなく、最後の MBConv ステージ 384-d 出力から蒸留
- スコアは同等で訓練 20% 高速化 → より多くの実験が可能

#### アンサンブル比率の OOF 最適化
- 現在 ProtoSSM 60% / SED 40% を固定値
- B1 SED / 純 CNN 追加後に OOF で比率を最適化
- 3-way blend: ProtoSSM / Tucker B1 SED / 純 CNN

---

## 後処理の改善候補

| 手法 | 現状 | 改善案 |
|---|---|---|
| Rank blend | ProtoSSM 60% + SED 40% | 3-way blend、OOF で比率最適化 |
| Gaussian smoothing | 適用済み | カーネル幅の調整 |
| Sonotype mirroring | 10 クラス | 類似種ペアの拡張 |
| Adaptive thresholding | 44 レア種 | Ghost species 28 種に特化チューニング |

**Ghost species（focal clip なし 28 種）**: ProtoSSM が soundscape から自動的に拾えている。中でも Southern Orange-legged Leaf Frog (517063) のみ mean_p=0.747 でやや低いため要確認。

---

## CPU 推論時間の見積もり（現状）

| ステップ | 時間 |
|---|---|
| Perch ONNX キャッシュビルド（59 訓練ファイル） | ~107〜143 秒 |
| ProtoSSM 学習 | ~10〜13 秒 |
| MLP probes 学習（58 クラス） | ~15 秒 |
| ResidualSSM 学習 | ~1〜2 秒 |
| Tucker SED 推論（B0 5-fold, 20 test files） | ~46〜67 秒 |
| **合計（dry-run）** | **~7〜12 分** |

B1 モデルは B0 より重いため、fold 数を増やす場合は推論時間を事前に確認する。
SED fold 数を 5 → 3 に削減すると推論 40% 削減可能だが精度低下あり。

---

## 優先アクション一覧

| # | アクション | 期待 LB 改善 | 工数 | ステータス |
|---|---|---|---|---|
| 1 | Tucker B1 fold 0〜4 完成 | +0.001〜+0.003 | 学習待ち | 🔄 進行中 |
| 2 | B1 SED を EoS5 に組み込んで LB 提出 | 確認 | 低 | ⬜ 待機 |
| 3 | 擬似ラベルで B1 SED 再学習 | +0.008〜+0.012 | 中 | ⬜ 待機 |
| 4 | 純 CNN（EfficientNetv2_s）訓練 → アンサンブル追加 | +0.002〜+0.005 | 中 | ⬜ 待機 |
| 5 | アンサンブル比率の OOF 最適化 | +0.001〜+0.002 | 低 | ⬜ 待機 |
| 6 | LSE Pooling 採用 | +0.005〜+0.010 | 高 | ⬜ 検討 |

---

## 参考: 調査で確認した主要 Discussion

| URL | 内容 |
|---|---|
| discussion/685318 | Tucker's Distilled SED 解説 (B0+Perch=0.898 vs B0 no Perch=0.876) |
| discussion/698538 | アンサンブル多様性の重要性（Perch系同士は+0.001のみ） |
| discussion/694479 | Gaussian smoothing の効果 |
| discussion/700763 | 純 CNN (EfficientNetv2) で 0.936 達成 |
| discussion/701711 | BirdNET が逆効果と確認 |
| discussion/701858 | Ghost species の分析 |
