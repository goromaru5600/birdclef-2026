# BirdCLEF+ 2026 振り返り

> 作成日: 2026-06-15  
> データソース: Kaggle API（リーダーボードCSV全件 + 上位50ノートブック）

---

## 最終リーダーボード（確定値）

| 指標 | 値 |
|------|-----|
| 総チーム数 | 4,092 |
| 評価指標 | ROC AUC（マクロ平均） |
| 1位スコア | **0.9672** |
| ゴールドボーダー（top 0.5% ≈ 21位） | 0.9579 |
| シルバーボーダー（top 5% = 204位） | 0.9513 |
| ブロンズボーダー（top 10% = 409位） | **0.9509** |

### Top 20

| 順位 | チーム | スコア |
|------|--------|--------|
| 1 | Nikita Babych | 0.9672 |
| 2 | Yannan Chen | 0.9670 |
| 3 | BirdCLEF+ 2026 Team 🤗 | 0.9633 |
| 4 | more exp is all you need | 0.9630 |
| 5 | ggkush - anilclaw | 0.9626 |
| 6 | YK | 0.9616 |
| 7 | BUET_Perceptron | 0.9607 |
| 8 | Arunodhayan | 0.9606 |
| 9 | Sinan Calisir | 0.9606 |
| 10 | Youssef Ouertani | 0.9603 |
| 14 | Takoi（日本人） | 0.9594 |

---

## 上位ソリューションの技術トレンド

Kaggle公開ノートブック（上位50件、投票数順）から抽出した技術要素。

### 1. Perch v2（Google）― ほぼ全員が使用

- Googleが公開した鳥音認識特化の音声基盤モデル（EfficientNet-B1ベース）
- スペクトログラム→埋め込みベクトルへの変換を担う
- 参考NB: `jaejohn/perch-v2-starter-train-infer`（295票）
- **教訓**: 専門ドメインの事前学習済みモデルは最初から使うべき。汎用モデル（EfficientNet等）から始めると差がつく

### 2. ProtoSSM（プロトタイプ型状態空間モデル）― 上位の核心技術

- 鳥の鳴き声を「時系列イベント」として扱うSequenceモデル
- Prototype（種の典型的鳴き声）を登録し、入力音声との類似度をSSM（State Space Model）でモデリング
- 参考NB:
  - `imaadmahmood/birdclef-2026-perch-v2-protossm-0-925`（412票、LB 0.925）
  - `hideyukizushi/bird26-reproduce-perch-protossm-resssm-inf-train`（272票）
- **教訓**: 音声は1次元時系列。CNNだけでなくSSM（Mamba系）やRNNで時系列パターンをモデリングするアプローチが強力

### 3. Distillationアプローチ（Pantanalモデル）― 最多投票NB

- Pantanaleデータセット（大規模鳥音データ）で事前学習したモデルから知識蒸留
- ONNXへの変換でKaggle Notebookの時間制約（9時間）に対応
- 参考NB: `dingjiarun/pantanal-distill-birdclef2026-onnx`（743票）
- **教訓**: 外部データ・事前学習モデルの活用がKaggleでは許可されることが多い。積極的に探すべき

### 4. SED（Sound Event Detection）― 補助的だが有効

- クリップ全体ではなく「イベント検出」として区間を特定してから分類
- Distillation + SEDの組み合わせが複数の上位解に登場
- 参考NB: `tuckerarrants/bc2026-distilled-sed`（264票）
- **教訓**: 鳥が鳴いていない区間（背景音）の影響を減らす工夫が精度に直結

### 5. EoS（Ensemble of Solutions）― nina2025の系列

- 単一モデルではなく複数の独立した解をアンサンブル
- nina2025はEoS.1〜EoS.9まで段階的に改善し、最終的に上位入賞
- 参考NB: `nina2025/birdclef-2026-eos-9`（598票）
- **教訓**: 多様なモデル（Perch + SED + ProtoSSM）の組み合わせが上位の定石。1モデルを極限まで磨くより複数の視点を組み合わせた方が安定する

### 6. PCEN（Per-Channel Energy Normalization）

- メルスペクトログラムの前処理として、従来の対数スケール変換より生物音響に適した正規化手法
- 参考NB: `pilkwang/birdclef-2026-eos-oof-gated-pcen`（219票）
- **教訓**: 音声のドメイン固有前処理（PCEN, dBFS正規化等）を軽視しない

### 7. 擬似ラベル（Pseudo-labeling）

- テストデータへの擬似ラベル付与 → 再学習のイテレーション
- LB 0.934 達成例あり
- 参考NB: `needless090/birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s`（223票）
- **教訓**: 終盤に擬似ラベルで伸ばすサイクルを組み込む余裕を最初から設計する

### 8. RankBlend

- 確率値のアンサンブルではなく、**順位**ベースのブレンド
- モデル間のスコールスケール差を吸収する
- 参考NB: `itshyao/birdclef-2026-s124-s114-g124-f1-rankblend`（257票）
- **教訓**: アンサンブル時はスコアをそのまま平均するのではなく、順位変換を検討する

---

## 今回の反省点（一般化）

### 技術的反省

| カテゴリ | 反省点 | 次回への教訓 |
|---------|--------|------------|
| **モデル選択** | 汎用CNNから始めてしまいがち | コンペ開始直後にドメイン特化の事前学習済みモデルを調査する |
| **時系列モデリング** | 画像分類として扱いがち | 音声・時系列データはSSM/RNNで時間依存性を明示的にモデリングする |
| **外部データ** | コンペデータのみで完結させようとする | 使用可能な外部データセット・事前学習モデルをDiscussionで早期把握 |
| **アンサンブル設計** | 終盤に慌てる | 最初から「複数の独立した解を作る」ことを計画する |
| **前処理** | デフォルト設定で進める | ドメイン固有の前処理手法（PCEN等）を初期に調査する |

### プロセス的反省

| カテゴリ | 反省点 | 次回への教訓 |
|---------|--------|------------|
| **公開NB活用** | 投票数の高いNBを早期にフォークして動かす | コンペ参加直後に上位NBのベースラインスコアを確認する |
| **Discussion調査** | 技術Discussionを見落としがち | API未対応のDiscussionはKaggle Web上で手動確認を習慣化 |
| **時間配分** | 実験が後半集中になりがち | 前半2週間でベースライン確立・後半でアンサンブルという計画を守る |
| **CV設計** | OOFスコアとLBの乖離を放置しがち | CV vs LB相関を早期に確認し、ずれがあれば原因を特定する |

---

## 上位公開ノートブック一覧（参照用）

| 投票数 | タイトル | 著者 | LBスコア相当 |
|--------|--------|------|-------------|
| 743 | pantanal-distill-birdclef2026-ONNX | Xie Xin / dingjiarun | - |
| 598 | BirdCLEF+ 2026 \| EoS.9 | nina2025 | - |
| 418 | pantanal-distill-birdclef2026-Improvement | Yusuf Murtaza | - |
| 412 | BirdCLEF+ 2026 — Perch v2 + ProtoSSM | Imaad Mahmood | 0.925 |
| 372 | BirdCLEF+ 26 \| Two-Pass SSM + Advanced PP | Maryna Borovska | - |
| 295 | perch_v2 starter: train + infer | g john rao | - |
| 272 | Bird26\|REPRODUCE\|Perch+ProtoSSM+ResSSM | yukiZ | - |
| 264 | BC2026 Distilled-SED | Tucker Arrants | - |
| 257 | BirdCLEF 2026 S124 S114 G124 F1 RankBlend | yao17 | - |
| 223 | BirdCLEF+ 2026 Iter-Pseudo Perch+SED LB 0.934 | 没落者 | 0.934 |

---

## 次回BirdCLEF系コンペへの優先アクション

1. **Perch v2（または後継モデル）のスターターNBを初日にフォーク**して動かす
2. **ProtoSSMの論文・実装**を事前に理解しておく
3. **Pantanal等の外部データ**の使用可否をルールで確認し、活用を計画に入れる
4. **PCEN前処理**を標準として採用する
5. **アンサンブル計画**を最初から設計（Perch系 + SED系 + ProtoSSM系の3種）
6. **ブロンズボーダーの目安**: 総チーム数の10%以内 → 今回なら 0.9509 以上

---

*このファイルはKaggle API（リーダーボードCSV全件 + 公開ノートブック50件）をもとに自動生成しました*
