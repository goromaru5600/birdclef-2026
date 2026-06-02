# 2026-05-30 作業履歴: bronze閾値の訂正 → EoS.9採用で0.950到達 → 密集帯脱出の検証(EffV2M probe)

## 開始時の状況
- 最高 LB 0.949（exp019）、bronze は「~0.955」と誤認、締切 2026-06-03
- 前日に OOF 検証基盤（`oof_scorer.py`）を構築済み

## 実施内容（時系列）

### 1. v20(EffV2M 16%)=0.947、v21(exp026 xSED 0.50/0.50)=0.948
- EffV2M を SED段に線形16%で混ぜた v20 は 0.947 と悪化（重み上げるほど下がる＝統合が壊している兆候）
- 検証セットは「SED 重みを上げろ」と単調に示したが、v21(0.50/0.50) は LB 0.948 に低下
- **結論：検証の SED 優位はリーク。** rare クラス(npos=1)で SED が AUC=1.000＝記憶の証拠。ProtoSSM 60% が正しい
- ※「v21=0.498」はタイポで実際 0.948（API で確認）

### 2. 🎯 bronze 閾値の重大な訂正：0.955 → **0.950**
- Bearer 認証で leaderboard を直接ダウンロードして実測
- **4094チーム、bronze ボーダー(上位10%=409位)= 0.950**。我々の 0.949 は約778位（上位19%）
- 「+0.006必要」と思っていたが**実際は +0.001〜0.002 で銅圏**だった

### 3. 🔑 上流 EoS.9 の発見と「丸ごと採用」
- 我々のフォーク元 = `nina2025/birdclef-2026-eos-9`。我々は EoS**5**、上流は既に **EoS.9**（4世代先）
- **EoS.9 = 0.950 = bronze ボーダー**（"Ensemble 0.950" NB は EoS.9 のバイト単位コピーと diff で確認）
- EoS5→EoS9 の追加：**Temporal-shift TTA**[0,±1,±2]、**genus-proxy mapping**、**family-head**、**taxonomy平滑化**(genus_α=0.15)、新 `bc2026-distilled-sed-public`、rank_aware_power=0.4
- Bearer 認証で EoS.9 のソース/メタを pull → 新 slug `gorubachohu/birdclef2026-eos9-adopt` に push（0.949フォークは温存）
- 実行 complete（636秒、90分制約に余裕）→ **submit → public LB 0.950 確定！**

### 4. 公開ノートブック・2025年上位解法の調査
- top-2% 2025：**Quantile-Mix α=0.5**（等重み）、過去データ事前学習+0.013、Silero-VAD、中央5秒擬似ラベル
- **mel-CNN 多チャンネル積み重ねは失敗(0.515)** ← 我々の EffV2M 失敗と一致
- 公開上位は全て Perch+ProtoSSM+SSM 系に収束（#1 pantanal-distill 734票も同系）

### 5. ⚠️ 0.950 の立ち位置：密集帯のコイントス
- **741チームが 0.950 に密集**。bronze ボーダー(409位)はこの密集帯の*真っ只中*
- private LB で再シャッフル → 0.950 の bronze 確率は**約1/3**（コイントス以下）
- 安全圏は「密集帯の上＝0.951+」（155チームのみ）

### 6. 密集帯を抜ける道の検証
- **公開NB同士のブレンドは無効と実測**：EoS.9 と perch_ensemble の per-class 順位相関 **0.9947**、rank-mean しても EoS.9 を下回る（弱い方に引っ張られるだけ）
- 手元の非Perchモデル：BirdNet 出力は**全クラス定数で壊れ**（AUC=0.5）、使えず
- leaderboard 照合で **hideyukizushi=0.956(24位)** を発見＝0.956 は達成可能だが非公開モデル由来の公算
- **結論：密集帯脱出には「相関の低い強モデル」が必須。** 唯一の筋＝CNN（EffV2M/ConvNeXt）を**最上位 rank ブレンド**（前回失敗は SED段線形だったため）

### 7. EffV2M probe を push（検証ステップ、submission消費なし）
- 自己完結ノートブック `birdclef2026-effv2m-probe` を作成・push
- train_soundscapes 全66ファイルに EffV2M 推論 → `submission_effv2m.csv` をダンプ
- 就寝時点で **running**。完了後に `analyze_effv2m.py` で自動分析予定

## 成果サマリー
| 項目 | 状態 |
|------|------|
| bronze 閾値の訂正 | 0.955→**0.950**（実測）|
| EoS.9 採用 | **LB 0.950 確定**（自己ベスト更新、暫定銅圏）|
| 0.950 の評価 | 密集帯=private で約1/3の賭け。0.951+ が必要 |
| 公開NBブレンド | 相関0.995で無効と実証 |
| EffV2M probe | push済・実行中（相関/blend検証待ち）|

## 翌朝の最初のアクション（probe 完了後・自動実行予定）
1. `analyze_effv2m.py` 実行 → **rank-corr(EoS.9, EffV2M)** と **top-level blend 効果**を測定
2. **判定**：
   - 相関低 かつ blend↑ → 本番版（EoS.9 + EffV2M 最上位rank）を submit、さらに ConvNeXt(0.9754) も
   - 相関高 or blend↓ → CNN多様性は死亡 → 0.950 確保＋最終2枠選択に集中
3. 最終2枠の選択：#1 EoS.9(0.950) + #2 最も毛色の違うもの

## 判明した Kaggle API 技（Bearer/KGAT_ token）
- leaderboard 実測：`GET /competitions/birdclef-2026/leaderboard/download`（zip CSV、全チーム）
- 公開NB取得：`GET /kernels/pull?userName=&kernelSlug=`（.blob.source=ipynb, .metadata=datasources）
- NB一覧：`GET /kernels/list?competition=birdclef-2026&sortBy=voteCount`
- 実行状態：`GET /kernels/status?userName=&kernelSlug=`
- 出力DL：`GET /kernels/output?...`（各fileに署名付きurl）
- 自分のsubmit履歴/スコア：`GET /competitions/submissions/list/birdclef-2026`
- 著者スコア照合：leaderboard CSV の TeamMemberUserNames

## 教訓
1. **前提を実測で疑え**：「bronze=0.955」の思い込みが戦略を歪めていた。実際は0.950で目前だった
2. **上流を見ろ**：自分のフォーク元(EoS.9)が答え(0.950)を持っていた。4世代遅れていた
3. **相関こそが多様性**：公開NBは全部0.995相関→混ぜても無駄。脱出には非Perch系が必須
4. **0.950は密集帯=賭け**：メダルには「集団の上」(0.951+)が要る

## 追記（probe完了後・夜間自動実行）: EffV2M は「多様だが弱すぎ」で不採用

probe 完了（train_soundscapes は実は **10,658ファイル**あり全件処理で時間超過）。190ラベル行で分析：

| 指標 | 値 |
|------|-----|
| EffV2M 単体 AUC | **0.697**（EoS.9 は 0.993）|
| rank-corr(EoS.9, EffV2M) | **0.31**（＝狙い通り多様）|
| top-level blend | **どの重みでも単調に悪化**（5%で既に↓）|
| ラベル整合 | 234/234（バグ無し、probeログで確認）|

**結論：仮説は半分当たり**。EffV2M は確かに非Perch系で「多様（相関0.31）」だが、**focal学習CNNは soundscape ドメインで弱すぎる（0.697）**＝ドメインシフトで撃沈。多様でも弱ければ混ぜると下がるだけ。ConvNeXt も同じ chain の focal 学習なので同様に失敗する公算 → **CNN多様性で密集帯を抜ける道は死亡**。

### 密集帯脱出・全施策の最終結論（すべて NEGATIVE）
1. パラメータ微調整 → 全て 0.948（死）
2. 公開NB同士のブレンド → 相関0.995で無効（死）
3. CNN多様性(EffV2M) → soundscapeで弱すぎ(0.697)、混ぜると悪化（死）

**安価で確実に 0.950 を超える道は、手持ち資産には無い**、が誠実な最終結論。0.956(hideyukizushi)は非公開モデル由来。

### 現実的な残りプラン
- **0.950(EoS.9)を確保**し、最終2枠を最適化（EoS.9 + 最も毛色の違うもの）
- 任意：密集帯のタイ崩れ対策に「重みを少しずらした≈0.950版」を1つ（期待値ではなく宝くじ的分散）
- 明確に上に出るには soundscape 頑健な多様モデルの自前学習（数日・高リスク。Perchが既に強い領域なので分は悪い）

## 参考ファイル
- EoS.9採用: `tmp/kaggle_push_eos9_adopt/`（ipynb, kernel-metadata, push_eos9.py, dryrun_out/）
- EffV2M probe: `tmp/kaggle_push_effv2m_probe/`（effv2m-probe.ipynb, build_probe_nb.py, push_probe.py, analyze_effv2m.py）
- 検証基盤: `tmp/kaggle_push_eos5/oof_scorer.py`
- メモ: `oof-validation-harness.md`, `eos9-upstream-bronze.md`, `project_birdclef2026.md`
