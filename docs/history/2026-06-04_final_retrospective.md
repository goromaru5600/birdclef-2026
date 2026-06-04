# BirdCLEF+ 2026 最終振り返り

## 最終結果

| 指標 | 値 |
|---|---|
| **Public LB** | **0.950**（313位 / 4,244チーム） |
| **Private LB** | **0.941**（963位 / 4,244チーム） |
| **Bronze cutoff（private）** | 0.942（約424位） |
| **結果** | **銅メダル獲得ならず**（0.001 不足） |
| **Public bronze cutoff** | 0.950（rank 424、全員同点） |

### Score Distribution（Public LB）

```
0.967:   2 teams  ← 優勝候補
0.963:   2 teams
0.952:  56 teams
0.951: 131 teams
0.950: 966 teams  ← 我々（313位）、公開上限で密集
0.949: 251 teams
```

Public で同スコア 0.950 の 966 チームが private で再評価され、
我々は 0.941 に落下（-0.009）。bronze cutoff の 0.942 にあと 0.001 届かなかった。

---

## コンペ概要

- **タスク**: パンタナル（ブラジル）の生物音響 234 クラス識別、macro-averaged ROC-AUC
- **制約**: Kaggle notebook CPU のみ、推論 90 分以内
- **期間**: 2026-03-11 ～ 2026-06-03 23:59 UTC
- **参加チーム**: 4,244 チーム

---

## 使用アーキテクチャ（EoS.9 パイプライン）

nina2025 チームの公開 notebook **EoS.9**（Public LB 0.950）をベースに採用。

### 全体フロー

```
音声ファイル（60秒）
    ↓ 5秒窓で 12 分割
Perch v2 ONNX（Google Bird Vocalization Classifier）
    → 1536次元 embedding + logits
    ↓
三分岐アンサンブル（direct blend）
    ├─ Model_22 (2.2%) : yukiZ branch
    │       ProtoSSM + ResidualSSM
    ├─ Model_51 (0.85%): Karnakbayev PSSM（中間出力）
    └─ Model_74 (96.7%): Karnakbayev PowerOptimization ← 支配的
            ↓
         xSED rank blend [0.60 proto, 0.40 SED]
         rank_aware_scaling (power=0.60)
         apply_prior (lambda=0.75)
         adaptive_delta_smooth (alpha=0.20)
         Gate 1/2/3
    ↓
Taxonomy Smoothing
    genus_α=0.15, class_α=0.05
    ↓
submission.csv
```

### 主要コンポーネント詳説

#### 1. Perch v2（基盤モデル）
- Google が約 10,000 種の鳥類で事前学習した音声エンコーダ
- ONNX 形式で CPU 推論
- 出力: 1536 次元 embedding + 234 クラス logits

#### 2. ProtoSSM / ResidualSSM（系列モデル）
- SSM（Selective State-Space Model）でウィンドウ間の時系列を学習
- 60 秒の 12 ウィンドウを通じた「鳥の鳴き声の継続性」を捉える
- ProtoSSM: prototype-based、ResidualSSM: 残差接続で安定化

#### 3. xSED rank blend
- ProtoSSM（プロト）と SED（Sound Event Detection）の 2 系統を percentile ランク空間でブレンド
- `pred = 0.60 * rank(p_proto) + 0.40 * rank(p_sed)`
- SED は Tucker Arrants の公開モデル（Tucker_SED_B1、ONNX）

#### 4. rank_aware_scaling（Power Optimization）
- `p_scaled = p^power`（power=0.60）
- ランク上位の予測値を相対的に強調し、スパースな陽性ラベルへの対応力を高める

#### 5. apply_prior（Bayesian 事前分布）
- 学習データから構築した「サイト×時刻×種」の出現頻度テーブルを事前分布として利用
- `p_final = (1 - λ) * p_model + λ * p_prior`（lambda=0.75）
- 未知の場所・時刻での出現が少ない種を抑制

#### 6. adaptive_delta_smooth
- 連続ウィンドウ間の予測値を平滑化（alpha=0.20）
- 瞬発的なノイズを抑え、持続的な鳴き声の予測を安定化

#### 7. Gate 関数（3 種）
```python
# Gate 1: ノイズ抑制（ProtoSSM 強 & SED 弱 → ProtoSSM 信頼）
fake_only = (p_proto > 0.50) & (p_sed < 0.05)

# Gate 2: 時系列継続性（Japanese Amendment 適用済み）
proto_cont = (xctx > 0.88) & (rank_proto > 0.77) & (p_sed < 0.14) & (~fake_only)

# Gate 3: SED スパイク保護（SED 強 & ProtoSSM 弱 → SED 信頼）
sed_only = (rank_sed > 0.95) & (rank_proto < 0.80) & (~fake_only) & (~proto_cont)
```

#### 8. Taxonomy Smoothing
```python
# Genus 内の種間で予測値を平滑化
p_c ← (1 - genus_α) * p_c + genus_α * mean_{j in genus}(p_j)

# Class 内でも同様
p_c ← (1 - class_α) * p_c + class_α * mean_{j in class}(p_j)
```
稀少種をその近縁種の予測で底上げする効果がある。

---

## 自分たちで取り組んだ工夫

### 1. EfficientNet-B0 の soundscape 学習（pseudo-label アプローチ）

**目的**: Perch と非相関な独立モデルを構築し、アンサンブル効果を狙う

**手順**:
1. EoS.9（0.950）で unlabeled soundscape 999 ファイルを推論し擬似ラベル生成
2. EfficientNet-B0 を soundscape mel-spectrogram で 5-fold 学習
3. Phase 3: 実際の soundscape ラベルファイルでの検証

**結果**:
- Val macro-AUC: **0.899**（focal 学習のみでは 0.697 → 大幅改善）
- しかし EoS.9 との rank 相関が高く、ブレンドしても **0.949** 止まり

**教訓**: ドメイン適応（soundscape 擬似ラベル）は有効だが、Perch エンベディングを共有する以上、独立性には限界がある。

### 2. パラメータ実験（10 本）

EoS.9 の後処理パラメータを 1 点ずつ変更して LB で検証。

| 実験 | 変更内容 | Public LB |
|---|---|---|
| eos9-adopt（ベース） | — | **0.950** |
| genus_α 0.20 | 0.15 → 0.20 | 0.950 |
| genus_α 0.10 | 0.15 → 0.10 | 0.950 |
| xSED [0.585, 0.415] | [0.60, 0.40] 変更 | 0.950 |
| delta_smooth 0.15 | 0.20 → 0.15 | 0.950 |
| lambda_prior 0.70 | 0.75 → 0.70 | 0.950 |
| lambda_prior 0.80 | 0.75 → 0.80 | **0.948**（悪化） |
| xSED585 + genus010 | 2 パラメータ同時変更 | 0.950 |
| xSED585 + genus020 | 2 パラメータ同時変更 | 0.950 |
| v3 blend（B0 w=0.05） | CNN ブレンド追加 | 0.949 |

**結論**: EoS.9 は Public LB で局所最適。全実験が中立か悪化。

### 3. 最終 2 枠選択

- 枠 1: `eos9-adopt`（ベースライン確実）
- 枠 2: `eos9-xsed585-genus010`（最も予測分布が異なる変種）

---

## なぜ Private で落ちたか（根本原因分析）

### Public 0.950 → Private 0.941（-0.009）の原因

| 要因 | 説明 |
|---|---|
| **apply_prior の過適合** | lambda=0.75 という強いサイト×時刻事前分布が Public テストの特性に最適化されていた。Private では異なる分布 |
| **Taxonomy smoothing の過適合** | genus_α=0.15 も Public LB 信号を基に選択。Private では最適でない可能性 |
| **Public テストセットのバイアス** | Public の 190 行はランダムサンプルではない可能性。EoS.9 はこの偏りに適合していた |
| **独立モデルの欠如** | Perch 依存の単一パイプライン。Private でも Perch の弱点をそのまま引き継ぐ |

### なぜ他のチームは Private で強かったか

| 手法 | 推定スコア帯 | 理由 |
|---|---|---|
| nina2025 EoS.10/11（private） | 0.951-0.960 | 非公開改善版（Private でも安定） |
| pilkwang PCEN sidecar | 0.951-0.955 | 独自 ConvNeXt + OOF-gated 補正。Private でも効く |
| 独自 GPU 学習モデル（上位陣） | 0.960-0.967 | Perch と非相関な強力なモデルで真のアンサンブル効果 |

---

## 学んだこと・次回への教訓

### 1. Public LB は過信禁物（最重要）
0.950 の 966 チームが private で大幅にシャッフルされた。Public 上限に到達したと思っても、private では全く別の競争になる。

### 2. 「強さ×独立性」フロンティアを早期に解決する
Perch エンベディングを共有する限り、どんなモデルも EoS.9 と高相関になる。次回は **別の音声基盤モデル**（BirdNET, AST, PaSST 等）や **mel-spectrogram CNN** を早期から並行開発すべき。

### 3. Pseudo-label の活用方針
今回の B0（val AUC 0.899）は良い出発点だった。次は：
- EoS.9 で全サウンドスケープ（～10,000 ファイル）を教師に拡大
- 信頼度フィルタで高品質な擬似ラベルに絞る
- 2 ラウンド擬似ラベリング（EoS.9 + CNN → 再擬似ラベル → 再学習）

### 4. Public LB 実験は方向確認のみ
10 本のパラメータ実験で「全部 0.950」は「EoS.9 が局所最適」ではなく「Public LB の分解能が足りない」可能性があった。OOF harness や private でのみ差が出るパラメータ変更は見逃していた。

### 5. 締切間際の情報収集
EoS.10/11（nina2025 private）が 0.951+ を達成していた。締切前日〜当日に公開される可能性を見越して、5 submission 枠を最大限活用する計画を立てておくべきだった。

### 6. prior の強さは private リスク
`apply_prior`（lambda=0.75）は強い事前分布で Public には効果的だが、Private との分布差に脆弱。lambda を下げる（0.65 等）実験は「中立」ではなく「private 安定性向上」の意味があったかもしれない。

---

## タイムライン

| 日付 | 内容 |
|---|---|
| 5月初旬 | EfficientNet-B0 学習、pseudo-label 生成開始 |
| 5/17-20 | EoS.5 採用、OOF harness 構築 |
| 5/25-28 | EoS.9 採用（0.950 確保）、Tucker SED blend 検討 |
| 5/29-30 | genus/delta/xSED パラメータ実験 |
| 6/1 | lambda_prior 実験（0.70=0.950, 0.80=0.948） |
| 6/2 | 合わせ技実験（xSED585+genus010/020）、最終 2 枠選択 |
| 6/3 23:59 UTC | **締切** |
| 6/4 | Private LB 公開：0.941（963位）、bronze 獲得ならず |

---

## まとめ

今回は公開パイプライン（EoS.9）の限界まで寄り切ったが、Private との乖離（-0.009）が銅メダルを阻んだ。EoS.9 は CPU 制約下での公開上限だが、Private では独自の多様なモデルを持つチームに大きく水を開けられた。次回は「Perch に依存しない独自モデルの育成」を最優先課題とする。
