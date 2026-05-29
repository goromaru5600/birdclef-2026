# 2026-05-29 作業履歴: v20結果分析 → OOF検証基盤の再構築 → exp026 (xSED SED寄り) push

## 開始時の状況

- **目標**: LB 0.955 (銅メダル)、締切 2026-06-03 (残り5日)
- **現状**: 最高 LB 0.949 (v12/exp019)。CPU-only 90分制約
- **アンサンブル構成**: ProtoSSM 60% + Tucker B0 SED 40% (cell 13 内で quantile-mix ブレンド)
- **直近の流れ**: EffV2M (EfficientNetV2M ONNX) 統合を試行中

## v20 (exp025) の結果報告: 0.947 (悪化)

EffV2M の最終寄与を ~16% に引き上げた v20 が **0.947** と、baseline 0.949 から下がった。

EffV2M の重みとスコアの関係:

| 実験 | EffV2M 寄与 | LB |
|------|------------|-----|
| exp019 (baseline) | 0% | **0.949** |
| v19 (exp024) | 6% | 0.948 |
| v20 (exp025) | 16% | **0.947** |

→ **重みを上げるほど悪化** = 単なるチューニング不足ではなく「統合が効いていない／壊している」典型的兆候。

## 実施内容

### 1. EffV2M ONNX の健全性チェック → モデル自体にバグ無し

`~/Downloads/effv2m_ckpt.onnx` を onnxruntime で直接検査:
- 入力: `(N, 128, 501, 1)` ✓ 前処理と一致
- 出力: `(N, 234)` sigmoid (0-1) ✓ スケール正常

→ 形状・スケールのバグは無い。悪化原因は (a) 窓長の前処理ミスマッチ (学習5秒/501frame、推論で SED chunks を501に切り詰め)、または (b) EffV2M が4モデル中最弱 (OOF 0.9738) を既調整済みの段に**線形で**混ぜたこと、に絞られた。

### 2. オフライン OOF 検証基盤の再構築 (最重要)

**発見**: EoS5 ノートブックには **dry-run フォールバック**がある。`test_soundscapes/` が空 (エディタ Run-All) のとき、cell 7 (ProtoSSM) と cell 13 (SED) が自動で `train_soundscapes/` を処理し、`BC2026_Train_*` の row_id で `submission_protossm.csv` / `submission_sed.csv` を書き出す。

これらの row_id は `data/train_soundscapes_labels.csv` と突合可能。`eval_data.npz` (P,S,Y,class_mask) の正体もこの dry-run 出力 (20ファイル×12窓=240行、うちラベルあり**190行**) と判明。

**作成**: `tmp/kaggle_push_eos5/oof_scorer.py`
- dry-run の submission CSV × ラベルで macro ROC-AUC をローカル計算
- → **Kaggle submission を消費せず無限に重み・ブレンド方式を実験可能**に
- 既存の `submission_*.csv` (946_fork) で `eval_data.npz` を再現確認

**検証ソースの上限**: train_soundscapes = 66ファイル / 1478セグメント / **75クラス** (234中)。159クラスは決して含まれない → 微調整は方向性のみ信頼、絶対値は信用しない (ランダムモデル birdnet を足すと42クラス上で見かけ上「改善」してしまうことを確認)。

### 3. 検証で判明した重要な事実

| 検証結果 | 含意 |
|---------|------|
| production は既に proto+sed を **quantile/rank ブレンド** (cell 13 `_qp`/`_qs`) | 「rank化」は無料の勝ち筋ではない (設計済み) |
| 検証で **SED 重みを上げるほど単調に AUC 改善** | SED を増やす方向が有望 |
| EffV2M は重み上げるほど悪化 (0.948→0.947) | 無効化が妥当 |

**quantile-mix proto_w スイープ** (190行/42クラス):

| proto / sed | rank-mix AUC |
|-------------|--------------|
| 0.20 / 0.80 | 0.99568 |
| 0.40 / 0.60 | 0.99468 |
| 0.50 / 0.50 | 0.99353 |
| **0.60 / 0.40 (現行 0.949)** | **0.99194** |
| 0.65 / 0.35 | 0.99124 |

**決定的な気づき**:
> これまでの xSED 実験 (exp021 の 0.65/0.35 など) は**全て「proto を増やす」方向**で全部 0.948 に下がった。**「SED を増やす」方向は LB で一度も試していない。** 検証セットはその逆方向 (SED 寄り) を強く・単調に示している。

### 4. exp026 (v21) push: xSED を SED 寄りに

**変更** (`fix_eos5_v21_exp026.py`、現行 v20 状態から):
1. `EFFV2M_W` 0.40 → 0.0 (EffV2M をクリーンに無効化 = exp019 baseline 復帰)
2. xSED `[0.60,0.40]` → `[0.50,0.50]` (Cell 2 solut + Cell 13 Model_5)

→ exp019 (0.949、EffV2M無し) から **xSED だけを変えた1変数実験**。

**結果**: push 成功 (versionNumber=21, kernelId=119788727, hasError=false)
- URL: https://www.kaggle.com/code/gorubachohu/birdclef2026-eos5-fork

## 成果サマリー

| 項目 | 状態 |
|------|------|
| v20 (EffV2M 16%) | LB **0.947** (悪化、確定) |
| EffV2M ONNX 健全性 | バグ無し確認 (悪化原因は統合方法/前処理) |
| OOF 検証基盤 `oof_scorer.py` | **構築完了** (submission 消費ゼロで実験可能に) |
| exp026 (v21, xSED 0.50/0.50) | **Pushed, 実行待ち** |

## v21 結果次第の判断ツリー

- **v21 ≥ 0.950**: SED方向が正解。さらに [0.40,0.60] へ (検証は単調改善示唆)
- **v21 = 0.949**: 横ばい。[0.40,0.60] で更にSED寄りを試す価値あり
- **v21 < 0.949**: 検証セット(42クラス)が誤誘導 → xSED 0.60/0.40 が最適と確定、リッチ検証ランで精査し直す

## 次のアクション

### ユーザー実行
1. **v21 を Run-All → Submit** して LB を確認
2. v20(0.947) を最終選択 submission にしない (0.949 を守る)

### アシスタント実行予定
1. **リッチ検証ランの構築**: dry-run を全 train_soundscapes (66ファイル/75クラス) に拡張し、proto/sed に加え **EffV2M も `submission_effv2m.csv` にダンプ**。以降の xSED 重み・EffV2M 寄与の精査を全てオフライン (75クラス、42より頑健) で実施
2. v21 LB 結果次第で xSED を更に SED 寄りへ

## 参考ファイル

- 修正済 notebook: `tmp/kaggle_push_eos5/birdclef-2026-eos-5.ipynb` (v21=exp026 状態)
- OOF スコアラー: `tmp/kaggle_push_eos5/oof_scorer.py` (再利用可能)
- 検証データ: `tmp/kaggle_push_eos5/eval_data.npz` (P,S,Y,class_mask) / dry-run CSV (`tmp/kaggle_push_946_fork/`)
- 実験スクリプト: `fix_eos5_v21_exp026.py` (本日)、v16〜v20 (過去)
- EffV2M ONNX: `~/Downloads/effv2m_ckpt.onnx` (入力 N×128×501×1, 出力 234 sigmoid)
- Push: `push_v9.py` (KGAT_ token + Bearer auth)

## 教訓

1. **blind submit でのパラメータ振りは完全に頭打ち** (lambda_prior/xSED/rank power/file_conf いずれも単独調整は 0.948)。0.949→0.955 はこの方法では埋まらない
2. **dry-run + ラベル突合で submission を消費せず検証できる** — 早くこれをルーチン化すべきだった
3. **検証セットは42→75クラスに拡張可能だが、それでも159クラスは欠落** → 方向性判断に使い、絶対値は信じない
4. **過去の実験は無意識に片方向 (proto増) ばかり試していた** — 検証で逆方向 (SED増) が未踏と気づけた
