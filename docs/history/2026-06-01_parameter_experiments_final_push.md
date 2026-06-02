# 2026-06-01 作業履歴: 最終パラメータ実験 × 4本 & 公開ノートブック調査

## 開始時の状況
- ベース: `eos9-adopt` public LB **0.950**（bronze line）確保済み
- v3 blend = 0.949（前日実験、harness 予測通り）
- BNhead blend = 中立（w=0 best）→ 0.950 が実質的な public 上限
- 残 2 日（締切 2026-06-03）

---

## 1. v3 blend 結果確認

`birdclef2026-eos9-v3blend`（B0 CNN w=0.05）= **0.949**（0.950 から微減）。
harness の +0.0001 予測通り、ノイズ範囲での下振れ。EoS.9 blend 実験は全滅確定。

---

## 2. 公開ノートブック調査（最新情報スキャン）

Kaggle API (`kernels/list?competition=birdclef-2026&sortBy=voteCount/scoreDescending`) で上位 30 本を精査。

### スコア降順で注目したもの

| ノートブック | 状況 |
|---|---|
| `nina2025/birdclef-2026-eos-9` | EoS.9 = 0.950 confirmed（公開上限） |
| `pilkwang/birdclef-2026-eos-oof-gated-pcen` | EoS.9 ベース + PCEN/BirdNET サイドカー → **private datasets 必須**（404）再現不可 |
| `thomaszyxu/bc26-v2538d-smart-hybrid` | 同上、pilkwang sidecar 依存 |
| `marynaborovska/birdclef-26-two-pass-ssm-advanced-pp` | Two-Pass SSM V16=0.924 → V17 で 11 改良。EoS.9 より低い |
| `karnakbaevarthur/hierarchical-taxonomy-pp` | 3 モデルブレンド（22+51+74）、0.950 ベース |

**結論: 0.950 を超える確認済み公開ノートブックは存在しない。**  
「スコア降順」ソートは著者の private 提出スコアを反映しており、ノートブック自体の確認スコアとは別物。

### PCEN サイドカーの評価

pilkwang は「OOF-gated rank correction」という手法で EoS.9 に PCEN/ConvNeXt モデルを局所的に補正している。
ただし必要な private datasets（`pilkwang/birdclef26-sidecar-exp001` 等）は 404 で取得不可。
自前で再現するには GPU 学習 + OOF 評価が必要で、2 日では困難と判断。

### ディスカッション注目スレッド（API から一覧取得）

| スレッド | c数 | 内容推定 |
|---|---|---|
| "How much did pseudo-labeling help you?" | 66 | 擬似ラベル有効性の議論 |
| "Is massive pretraining essential now?" | 10 | 大規模事前学習の必要性 |
| "What's the limit without any Perch" | 35 | Perch 非依存の上限 |

---

## 3. 古い提出ノートブックのアンサンブル可能性を検討

ユーザーの提案: `perch-protossm-sed-946`、`perch-efficientnet-ensemble` などを ensemble できないか？

調査結果:
- 古い Perch ベース（946 等）: 同じ backbone → 相関 ~0.99 → ブレンド効果ゼロ
- 自前 EfficientNet: 5 月中旬の初期モデル → soundscape AUC ~0.70 推定 → 同じ「独立だが弱い」問題
- `tucker-sed-b1` / `pseudo-labels-generator`: EoS.9 に組み込み済み or 推論ノートブックではない

**結論: 既存提出物のアンサンブルも有効な多様性を提供できない（強さ↔独立性フロンティアの再確認）。**

---

## 4. パラメータ実験 × 4 本を設計・push・submit

EoS.9 (0.950) ベースで未試験のノブを 1 行変更ずつ試す方針。

### 変更箇所の確認

Cell 30（taxonomy smoothing）:
```python
def f_TAX_SMOOTHING_POSTPROC(func_add=direct, genus_α=0.15, class_α=0.05):
```
Cell 23（Model_74 Karnakbayev branch）:
- `rank_aware_scaling(power=0.6)`
- `apply_prior(lambda_prior=0.75)`
- `adaptive_delta_smooth(base_alpha=0.20)`
- xSED `[0.60, 0.40]`（Cell 4 solutions dict）

### 4 本の実験

| slug | 変更 | 理由 |
|------|------|------|
| `eos9-tax-genus020` | genus_α **0.15→0.20** | 稀少種に genus smoothing 強化 |
| `eos9-tax-genus010` | genus_α **0.15→0.10** | A の逆方向（A 失敗時の保険） |
| `eos9-xsed-0585` | xSED **[0.60,0.40]→[0.585,0.415]** | Karnakbayev power-opt 推奨値 |
| `eos9-delta-smooth-015` | delta alpha **0.20→0.15** | 瞬発的な鳴き声を過平滑化しない |

全て 2026-05-31 深夜に Kaggle 上で完走。2026-06-01 に 4 本全て submit。

---

## 5. 組み合わせ実験の設計論

ユーザーの質問「掛け合わせは必要?」に対する分析:

- 方向不明な状態で組み合わせ → 期待値変わらず、分散増加
- 独立パラメータ（genus α と xSED）は同日 submit で並行情報収集が最適
- 正しい順序: 単独確認 → 正方向と判明したら組み合わせ

---

## 成果サマリー

| 項目 | 状態 |
|------|------|
| ベース | **0.950 確保**（eos9-adopt、無傷） |
| 公開 NB 調査 | 0.951+ の確認済み公開 NB なし |
| 古い提出アンサンブル | 有効な多様性なし（結論済み） |
| 新実験 | A/B/C/D × 4 本 submit 済み、結果待ち |
| 残タスク | 結果確認 → final 2 枠選択（締切 6/3） |

## Final 2 枠選択指針

1. `eos9-adopt` (0.950) は必ず 1 枠
2. A/B/C/D で 0.951+ が出ればそれを 2 枠目
3. 全部 0.950 なら最もパラメータが異なるもの（xsed585 推奨）
4. 全部 0.949 なら `eos9-adopt` を 2 枠（同一でも可）

## 教訓

1. **「スコア降順」API ソートを信用しない** — ノートブック自体のスコアではなく著者の最高 LB スコアを反映
2. **古い提出物のアンサンブルは新たな多様性を生まない** — 全部 Perch ベースで高相関
3. **パラメータ実験は締切 2 日前でも有効** — 1 行変更・既成コード流用でゼロコスト
