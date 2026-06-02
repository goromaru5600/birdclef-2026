# 2026-05-31 作業履歴: blend実験の網羅的探索 → 「強さ↔独立性」フロンティアの発見 → BirdNET native

## 開始時の状況
- 前日 EoS.9 を丸ごと採用し **public LB 0.950**（bronze ライン）到達、ベース確保済み
- 課題: 0.950 は **741チーム密集帯**。private シェイクで脱落リスク → **0.951+ で密集帯を抜けたい**
- 方針: 「EoS.9 と非相関で、かつ強いモデル」を rank-blend で足せば抜けられるはず

## 核心の発見: 「強さ ↔ EoS.9独立性」はトレードオフ（両立不可）

「強い×独立」なモデルを作ろうと、6+モデルを網羅的に学習・検証（全て val_preds → harness で corr & blend ゲート）:

| モデル | soundscape AUC | rank-corr(EoS.9) | best blend | 備考 |
|--------|------|------|------|------|
| v3 b0 純ソフト蒸留 | 0.87 | 0.55 | +0.0001 | |
| v2 b0 focal事前+蒸留 | 0.88 | 0.59 | 0 | **実LB submit=0.949(↓)** |
| nfnet eca_nfnet_l0(別アーキ) | 0.88 | 0.59 | 0 | アーキ変えても相関同じ |
| focalheavy(focal比率↑) | 0.86 | 0.55 | 0 | |
| hardlabel(擬似ラベル二値化) | 0.77 | 0.39 | 0 | corr下がるが弱体化 |
| BirdNET probe(focal学習) | 0.72 | 0.30 | 0 | 独立だが弱い |
| BirdNET V3 native | 0.66* | 0.56 | +0.00009 | *harness18/42しかカバー |
| EffV2M focal CNN | 0.70 | 0.31 | 0 | |
| 独立3つの平均 | 0.79 | 0.43 | 0 | 束ねても強さ不足 |

**メカニズム（実証済み）:**
- EoS.9 を教師に蒸留 → 強いが相関する（~0.59、**アーキ非依存**: nfnet==b0）
- 独立に学習（focal/BirdNET）→ 真に非相関(0.30)だが soundscape で弱い(~0.70)
- hardlabel が両者を繋ぐ証拠: corr 0.59→0.39 に下がるが AUC 0.88→0.77 に低下
- **EoS.9 が soundscape で 0.99 と強すぎ**て、弱い独立モデルでは原理的に勝てない

→ **手持ちで「強い×独立」は作れない**ことが理論・offline・実LBの三方向で確定。

## 実LB での裏付け
- **v2 CNN blend (w=0.05) = 0.949**（0.950から↓）。harness の中立判定が実LBで正しいと確認（リーク過小評価ではなかった、このケースでは）

## ハーネスの構造的限界（最大の学び）
- 検証 harness = 42クラス・SEDリークで EoS.9 が満点・BirdNETは18/42しかカバー
- → **独立追加を構造的に過小評価**。「全部中立」は一部 harness の artifact
- → **実テストの rare/非リーククラスでの効果は harness では永遠に見えない** → 実LBが唯一の真の判定

## BirdNET 深掘り（規約・ラベル・native）
- **規約確認**(ユーザー指摘): 外部事前学習モデルはホストがPerchを公式提供＝許可。BirdNET V3.0 = **CC BY-SA 4.0（商用OK）**、禁止は密猟/軍事のみ、bronzeはコード提出不要でShareAlike無関係 → **使用可**
- **モデル**: `nehpadvi/birdnet-v3-0` の BirdNET_V3.0-preview3_Global_11K ONNX（predictions[11560]、入力32kHz可変長=競技と一致）
- **ラベル**: Zenodo record 18247420 から取得（`BirdNET+_V3.0-preview3_Global_11K_Labels.csv`、`;`区切り）
- **クラス一致**: **190/234（全162鳥 + 両生22 + 哺乳3 + 昆虫3）**
- **バグ#1発見**: BirdNETに5s入力（学習は3s）→ **3sチャンクに修正**

## 最終弾: EoS.9 + BirdNET 鳥クラス限定 per-class blend
- `birdclef2026-eos9-birdnet-birdblend` を push（鳥162クラスのみ w=0.05 rank-blend、非鳥はEoS.9のまま保護、写像162種を埋め込み、3s修正済み、internet OFF）
- dry-run 完走（NaNは既知のEoS.9 dry-run artifact、実submitでは正常）
- **submit待ち**（実LB判定: harnessが見えないrare bird効果が出るか）

## 成果サマリー
| 項目 | 状態 |
|------|------|
| ベース | **0.950 確保**（eos9-adopt, 無傷, public rank~191=銅圏先頭）|
| blend実験 | 9モデル全て harness中立、v2実LB=0.949 |
| 結論 | 手持ちで0.950超えは構造的に困難（強さ↔独立フロンティア）|
| BirdNET bird-blend | push済・submit待ち（最後の実LB賭け）|

## 自己批判（実装ミス・伸びしろ）
1. **BirdNET 5s→3s**（明確なバグ、修正済み）
2. **harnessのリーク・狭さ**（検証手法の限界。独立追加を過小評価）
3. 擬似ラベル教師が旧v17版（軽微）
4. **EoS.9内部ノブ（rank_power/Model重み/tax α）を実LB未調整**（±0.001の余地、未消化）
5. CNN推論にTTA未付与（軽微）

## 残タスク（コンペは継続中、締切2026-06-03）
1. **BirdNET bird-blend の実LB結果**を待つ
2. **最終2枠の選択**: #1 必ず eos9-adopt(0.950)、#2 最も毛色の違うもの
3. 締切まで**公開LB定期チェック**（誰かが>0.950公開したら乗り換え＝唯一の楽勝候補）
4. 余力あれば伸びしろ#4（EoS.9内部ノブ実LB調整）を1-2枚gamble

## 参考ファイル
- 学習: `tmp/kaggle_train_ssc_{b0,b0_v2,nfnet,hardlabel,focalheavy}/`
- BirdNET: `tmp/kaggle_birdnet_probe/`（probe/native）, `tmp/bn_v3_labels.csv`
- 最終弾: `tmp/kaggle_push_eos9_cnn/birdclef-2026-eos-9-birdnet.ipynb`, `birdnet_birdmap.json`
- 検証基盤: `tmp/kaggle_push_eos5/oof_scorer.py`
- メモ: `eos9-upstream-bronze.md`（全実験マトリクス記録済み）

## 教訓
1. **公開の強モデル(EoS.9)に「足して」勝つのは難しい** — 上位陣は自前で強い多様モデルを月単位で作っている。数日では再現不能
2. **リークありの狭い検証は独立追加を過小評価する** — 真の判定は実LBのみ
3. **相関はアーキでなく教師で決まる** — 別backboneは無駄、独立信号(別基盤モデル)が必要だが、それは弱い
4. **ベースを別kernelで温存**したのは正解 — 全実験が「下振れゼロの上振れ狙い」にできた
