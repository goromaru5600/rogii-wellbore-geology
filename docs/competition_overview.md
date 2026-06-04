---
marp: true
theme: default
paginate: true
style: |
  section {
    font-size: 22px;
    font-family: 'Helvetica Neue', sans-serif;
  }
  h1 { color: #1a4f7a; border-bottom: 2px solid #1a4f7a; }
  h2 { color: #2c7ab3; }
  table { font-size: 18px; }
  code { background: #f0f4f8; }
  .highlight { background: #fff3cd; padding: 4px 8px; border-radius: 4px; }
---

# ROGII - Wellbore Geology Prediction
## Kaggle Featured Competition 概要

- **主催**: ROGII（石油・ガス向け地科学ソフトウェア企業）
- **賞金**: $50,000
- **エントリー締切**: 2026-07-29
- **最終提出締切**: 2026-08-05
- **参加チーム数**: 2,267チーム
- **提出方法**: Kaggle Notebook 限定（コード提出）
- **メダル対象**: ✅

---

# 背景 - ジオステアリングとは

水平掘削において、**ドリルビットが狙った地層の中を正確に進み続けるよう制御する**技術

```
地表
 |
 |  垂直部分（縦に掘削）
 |
 └─────────────────────────────→  水平部分（横に掘削）
         ↑ この水平部分で石油・ガスが採掘される
         ↑ 地層から外れないよう TVT をリアルタイムで把握する必要がある
```

- 従来: 地質学者が手動でリアルタイム解釈 → **コスト・時間がかかる**
- 本コンペ: **機械学習で自動化**することが目標

---

# タスク定義

## 予測対象: TVT（True Vertical Thickness）

水平井の各1フィート地点における**地層内の垂直位置**を予測する回帰タスク

```
typewell（縦井 = 地層の基準）
    │  TVT=11300 → ANCC層
    │  TVT=11400 → ASTNU層
    │  TVT=11500 → EGFDU層
    │
    └──── horizontal well（横井）
              MD=11500ft → TVT=? を予測
```

- **入力**: 測定深度・座標・GRログ・初期TVT推定値
- **出力**: 各測定点の TVT 値（フィート単位）
- **評価指標**: MSE（Mean Squared Error）← 低いほど良い

---

# データ構造

## ファイル構成

```
data/
├── train/          # 773本の学習用 well
│   ├── {id}__horizontal_well.csv   # 水平井のセンサーデータ（正解ラベルあり）
│   ├── {id}__typewell.csv          # 縦井の基準データ（地層ラベルあり）
│   └── {id}.png                    # 地質断面図（画像）
├── test/           # 3本のテスト用 well（予測対象）
│   ├── {id}__horizontal_well.csv
│   └── {id}__typewell.csv
└── sample_submission.csv           # 提出フォーマット（14,151行）
```

- train: **773 wells** × 平均 6,800行/well
- test: **3 wells**、計 14,151点の TVT 予測が必要
- 提出ID形式: `{well_id}_{row_index}`

---

# 特徴量の説明（horizontal_well）

| カラム | 説明 | 学習時 | テスト時 |
|--------|------|--------|----------|
| `MD` | 測定深度（ft）: 掘削した管の長さ | ✅ | ✅ |
| `X`, `Y` | 水平座標（Easting/Northing） | ✅ | ✅ |
| `Z` | 垂直深度（負の値 = 地下） | ✅ | ✅ |
| `GR` | ガンマ線（API単位）: 地層の自然放射能 | ✅ | ✅ |
| `TVT_input` | 地質家が推定した初期TVT値 | ✅ | ✅ |
| `ANCC` | 地層境界の深度（ANCC層トップ） | ✅ | ❌ |
| `ASTNU`, `ASTNL` | 地層境界の深度（ASTNU/ASTNL） | ✅ | ❌ |
| `EGFDU`, `EGFDL` | 地層境界の深度（EGFDU/EGFDL） | ✅ | ❌ |
| `BUDA` | 地層境界の深度（BUDA層） | ✅ | ❌ |
| **`TVT`** | **予測対象（正解ラベル）** | ✅ | ❌ |

---

# 特徴量の説明（typewell）

縦井（typewell）は水平井と紐付いた**地層の基準ログ**

| カラム | 説明 |
|--------|------|
| `TVT` | 縦井の各深度での TVT 値 |
| `GR` | ガンマ線ログ（縦方向の参照） |
| `Geology` | 地層ラベル（22種類） |

## 主要な地層ラベル（Geology）

| ラベル | 出現頻度 | ラベル | 出現頻度 |
|--------|----------|--------|----------|
| ANCC | 最多 | EGFDL | 多 |
| ASTNL | 多 | BUDA | 多 |
| ASTNU | 多 | EGFDU | 中 |

---

# コアチャレンジ

## なぜ難しいか

テストデータには**地層境界カラム（ANCC, ASTNU等）が存在しない**

```
学習時: MD, X, Y, Z, GR, TVT_input, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA → TVT
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                     これがテストでは使えない！

テスト時: MD, X, Y, Z, GR, TVT_input のみ → TVT を予測
```

## 解決のアプローチ方向性

1. **typewell との GR パターンマッチング**でテスト well の地層境界を推定
2. **TVT_input を起点**にした補正モデルの構築
3. **well 間の座標近傍性**を利用した類似 well 検索

---

# 評価指標と現在のスコア

## 評価指標: MSE（Mean Squared Error）

$$MSE = \frac{1}{N}\sum_{i=1}^{N}(TVT_i - \hat{TVT}_i)^2$$

**RMSE換算での現在のリーダーボード（上位100件）**

| 順位 | スコア（RMSE相当） |
|------|-------------------|
| 1位 | **5.986** |
| 10位 | 7.723 |
| 20位 | 7.908 |
| 50位 | 8.170 |
| 100位 | 8.492 |

---

# メダルボーダーの試算

参加チーム数: **2,267チーム**（2026-06-05時点）

| メダル | 条件 | 推定必要順位 | 目標スコア（RMSE目安） |
|--------|------|-------------|----------------------|
| 🥇 Gold | 上位 ~10チーム（0.44%） | ~10位以内 | < 7.7 |
| 🥈 Silver | 上位 ~45チーム（2%） | ~45位以内 | < 8.0 |
| 🥉 Bronze | 上位 ~227チーム（10%） | ~227位以内 | < 8.5〜 |

> 現在の100位スコアが 8.492 なので、**ベースラインを超えれば銅メダル圏内の可能性あり**

---

# 提出の注意事項

## Kaggle Notebook 限定提出

- ローカルで学習 → **Kaggle Notebook に推論コードをアップロードして Submit**
- 1日最大 **5回** の提出制限
- 推論時間・リソース制限あり（Kaggle の GPU/CPU 環境）

## 推奨ワークフロー

```
ローカルで EDA・モデル開発
     ↓
Kaggle Notebook に推論スクリプトをアップ
     ↓
Notebook を実行 → submission.csv を出力
     ↓
Submit（5回/日）
```

---

# アプローチのアイデア

## Step 1: ベースライン
- `TVT_input` をそのまま予測値として提出（スコア確認）
- 単純な線形回帰 / LightGBM

## Step 2: typewell 活用
- 水平井の GR パターン ↔ typewell の GR パターンを DTW やコサイン類似度でマッチング
- typewell の地層境界を水平井に投影し、擬似的な地層境界特徴量を生成

## Step 3: 高度化
- 近傍 well（座標的に近い学習 well）から地層情報を転用
- 時系列モデル（LSTM / Transformer）で深度方向の連続性を学習
- マルチタスク学習（TVT + Geology ラベル予測の同時学習）

---

# タイムライン

| 日付 | マイルストーン |
|------|---------------|
| 2026-06-05 | 参加登録・データ取得完了 ✅ |
| 〜2026-06-中旬 | EDA・ベースライン構築 |
| 〜2026-07-中旬 | typewell 活用モデルの構築・改善 |
| 2026-07-29 | **エントリー締切** |
| 2026-08-05 | **最終提出締切** |

## 目標
- まず **ベースライン提出** でリーダーボードに乗る
- 最終的に **銅メダル（上位10%）** を獲得する

---

# 参考リンク

- [Kaggle コンペページ](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
- [公開ノートブック - Super Baseline (LB: 12.602)](https://www.kaggle.com/code/romantamrazov/rogii-super-baseline-lb)
- [公開ノートブック - Better Solution (LB: 9.956)](https://www.kaggle.com/code/romantamrazov/rogii-better-solution-lb-9-956)
- [公開ノートブック - DWT-based (LB: 9.251)](https://www.kaggle.com/code/nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based)
- [公開ノートブック - LightGBM Baseline](https://www.kaggle.com/code/pengchzn/rogii-lightgbm-baseline)
- [GitHub - argon approach](https://github.com/aaryan2203/rogii-wellbore-geology-prediction-argon)
