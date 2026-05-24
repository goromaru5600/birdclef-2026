# 🌅 朝チェックリスト (2026-05-25)

## 📊 現在のスコア状況

| Version | 内容 | LB | 状態 |
|---------|------|-----|------|
| v5 | B0 SED 5fold + ProtoSSM (rank-blend) | **0.949** | 基準値 |
| v6 | + B1 fold[1,2] @ 10% (古い学習) | 0.948 | -0.001 失敗 |
| v7 | B1除去 (v5復帰) | (推定 0.949) | Submit任意 |
| v8 | Quantile-Mix blend | **0.949** | 変化なし |
| **v9** | **+ B1 fold[0,3,4] Phase 2再学習 @ 10%** | **未測定** | **要Submit→LB確認** |

**目標 LB**: 0.955 (銅メダル) | **締切**: 2026-06-03 (残り9日)

---

## 🎯 朝一でやること

### Step 1: v9 Submit (まだなら)
1. https://www.kaggle.com/code/gorubachohu/birdclef2026-eos5-fork
2. Version 9 → Save & Run All 完了確認
3. **Submit to Competition**
4. LB結果待ち (1-2時間)

### Step 2: v9 LB結果による分岐判断

| v9 LB | 判断 | 次手 |
|-------|------|------|
| **≥ 0.952** | B1機能 ✅ | **lambda_prior 微調整でv10** (10分、+0.001狙い) |
| **0.950〜0.951** | 微改善 | **真の疑似ラベル再学習着手** (1日、+0.005〜0.010狙い) |
| **0.949** | 変化なし | **真の疑似ラベル再学習** or **純CNN追加** |
| **< 0.949** | B1悪化 | **v9 → v8相当にrevert**、別アプローチ |

---

## 🔧 ストック準備済み施策

### A. 真の疑似ラベル再学習 (最有力、+0.005〜0.010期待)

**前回失敗の原因**: 疑似ラベル生成対象が既ラベルfilesと重複 → `is_pseudo_mask.sum()=0`

**修正方針**:
- pseudo-labels.ipynb を修正: `train_soundscapes_labels.csv` に**含まれないファイル**を選択
- もしくは label外の chunk (file内の未ラベル時間帯) を対象
- Kaggle再生成 → Colab再学習 (12-15h)

**着手前ステップ**:
1. v9 LB確認
2. 疑似ラベルスクリプト修正 (30分)
3. Kaggle生成 (30-60分)
4. Colab再学習 (12-15h)
5. v10 push

### B. lambda_prior 微調整 (即効性)

現在 `lambda_prior=0.5`、試す価値:
- v10a: 0.55
- v10b: 0.60

Submit枠が許す範囲で1つ試す。

### C. Silero-VAD 人声除去 (未着手)

期待値は不明だが、2025上位がほぼ全員採用。中工数 (2-3h)。

---

## 📚 今日やったこと (要約)

### 完了
1. ✅ **v7 push**: B1除去でv5復帰版 (Submit任意)
2. ✅ **B案グリッド探索**: OOFリーク懸念で見送り判断
3. ✅ **C案 threshold tuning**: 既実装と判明、スキップ
4. ✅ **A案 公開Notebook調査**: subagent並行実行、7施策発見
5. ✅ **疑似ラベル生成 v2** (Kaggle): 820s完了、CSV取得
6. ✅ **v8 push** (Quantile-Mix): LB 0.949 (効果なし、下流pipeline吸収)
7. ✅ **Phase 2 B1再学習** (Colab): fold 0/3/4 完了、fold 0: 0.71→0.75
8. ✅ **3 ONNX を Kaggle dataset upload** (user手動)
9. ✅ **v9 push**: B1 fold[0,3,4] @ 10% で組込み
10. ✅ **Site/Hour priors調査**: 既実装と判明、スキップ

### 重要な学び
- **OOF評価は pipeline 末端 (`submission.csv`) で行うべき** — blend直後では下流が吸収
- **Subagent推奨手法も既実装verify必須** — Quantile-Mix も Site/Hour priors も既実装だった
- **+0.001以下のOOF差は信頼しない** — 評価誤差と区別困難
- **疑似ラベルは「既ラベルにないファイル」で生成すべき** — 重複だと効果ゼロ

### 失敗
- v6 (B1 fold[1,2] 古い学習): -0.001
- v8 (Quantile-Mix): +0.000 (期待+0.001~0.002)
- Phase 2 疑似ラベル: 実質再学習のみ、fold 0 改善は副産物

---

## 🔗 参照URL

| 項目 | URL |
|------|-----|
| Kaggle EoS5 notebook | https://www.kaggle.com/code/gorubachohu/birdclef2026-eos5-fork |
| Kaggle 疑似ラベル生成 | https://www.kaggle.com/code/gorubachohu/birdclef2026-pseudo-labels-generator |
| B1 ONNX dataset | https://www.kaggle.com/datasets/gorubachohu/560-sed-distill-fold0 |
| GitHub repo | https://github.com/goromaru5600/birdclef-2026 |
| Colab B1学習notebook | `tmp/colab_tucker_sed_b1/bc2026-sed-colab-pseudo-v1.ipynb` |

## 📂 ファイル位置

| ファイル | 用途 |
|---------|------|
| `tmp/kaggle_push_eos5/birdclef-2026-eos-5.ipynb` | EoS5 inference (v9) |
| `tmp/kaggle_push_eos5/push_v9.py` | Push script |
| `tmp/kaggle_push_pseudo/pseudo-labels.ipynb` | 疑似ラベル生成 v2 |
| `tmp/colab_tucker_sed_b1/bc2026-sed-colab-pseudo-v1.ipynb` | B1再学習 (Colab) |
| `docs/history/2026-05-24_v7_v8_quantile_mix_pseudo_labels.md` | 今日の詳細履歴 |

## 💤 おやすみなさい

明日 v9 の LB 結果を見て、次の一手を決めましょう。
