---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
  }
  table {
    font-size: 0.85em;
  }
  code {
    font-size: 0.8em;
  }
---

# CatBoost vs XGBoost
## 速度・精度・使い勝手の比較

Kaggle rogii-wellbore-geology コンペ文脈での整理

---

## アルゴリズムの違い

| | XGBoost | CatBoost |
|---|---|---|
| 木の構築 | Level-wise (深さ優先) | **Symmetric tree** (対称木) |
| 分割探索 | 全特徴量を逐次スキャン | GPU並列 / 対称構造で効率化 |
| カテゴリ変数 | エンコード必要 | **ネイティブ対応** |
| 勾配計算 | 標準勾配ブースティング | **Ordered Boosting**（リーク防止）|

> CatBoost の Symmetric tree = 全ノードで同じ分割条件 → 予測が速い・過学習しにくい

---

## 学習速度の比較（CPU）

```
データ: ~380万行 × 57特徴量 (rogii 訓練セット)
GroupKFold 5fold × 1モデル
```

| モデル | iterations | 目安時間 (CPU) |
|---|---|---|
| XGBoost | 1,000 | 約 5〜10 分 |
| LightGBM | 8,000 | 約 15〜25 分 |
| **CatBoost** | **8,000** | **約 30〜60 分** |

**CatBoost は XGBoost の 5〜10倍 時間がかかる**
→ CPU では特に遅い。GPU があれば 3〜5倍程度に縮まる

---

## なぜ CatBoost は遅いのか

### 1. Ordered Boosting
- 各サンプルの勾配を「それより前のサンプルだけ」で計算
- リーク防止のため計算量が増える

### 2. 対称木 (Symmetric Tree)
- 精度を出すには XGBoost より多くの木が必要になりやすい
- iterations=8,000 は XGBoost の n_estimators=1,000 に相当することも

### 3. カテゴリ変数の内部処理
- 数値のみのデータでも内部オーバーヘッドがある

---

## GPU での比較

| | CPU | GPU |
|---|---|---|
| XGBoost | baseline | 5〜10x 高速 |
| LightGBM | 速い | 3〜5x 高速 |
| **CatBoost** | 遅い | **10〜50x 高速** ← GPU 恩恵が最大 |

> CatBoost は GPU で最も恩恵を受けるモデル
> Kaggle GPU (T4) を使えば 30〜60 分 → 5〜10 分 に短縮可能

---

## 精度の傾向

| シナリオ | 有利なモデル |
|---|---|
| カテゴリ変数が多い | **CatBoost** |
| 数値特徴量のみ | LightGBM ≈ XGBoost |
| 小〜中規模データ | **CatBoost** (Ordered Boosting でリーク防止) |
| 大規模データ (1000万行+) | LightGBM |
| アンサンブルの多様性 | CatBoost + LGB の組み合わせが有効 |

rogii の場合: 数値特徴量のみ → **LightGBM と精度差は小さい**
アンサンブルの多様性確保が主な目的

---

## rogii Phase 2-4 での構成

```python
# LightGBM × 3 seeds  (speed重視、多様性確保)
LGB_CONFIGS = [
    dict(learning_rate=0.025, n_estimators=8000, seed=42),
    dict(learning_rate=0.020, n_estimators=8000, seed=7),
    dict(learning_rate=0.030, n_estimators=8000, seed=123),
]

# CatBoost × 1  (多様性の追加、精度補完)
CB_PARAMS = dict(
    iterations=8000, depth=7, od_type='Iter', od_wait=300
)

# Ridge Stacking で重み最適化
ridge = Ridge(alpha=1., positive=True)
```

総計: **4モデル × 5fold = 20モデル** → Kaggle 9h タイムアウトに注意

---

## タイムアウトリスク

Kaggle notebook の制限: **9時間**

| 処理 | 推定時間 |
|---|---|
| データ読み込み + 特徴量生成 | ~20 分 |
| LGB × 3 seeds × 5 fold | ~75〜125 分 |
| CatBoost × 5 fold | ~150〜300 分 |
| **合計** | **~4〜7 時間** |

ギリギリのライン。もしタイムアウトするなら:
- CB の `iterations` を 5,000 に下げる
- CB を抜いて LGB × 3 のみにする

---

## まとめ

| 観点 | 結論 |
|---|---|
| 速度 | XGBoost > LightGBM > CatBoost (CPU) |
| GPU 恩恵 | CatBoost が最大 |
| 精度 (数値特徴量) | LightGBM ≈ CatBoost > XGBoost |
| アンサンブル効果 | LGB + CB の組み合わせが鉄板 |
| rogii での用途 | 多様性確保 + 精度補完 |

> **Phase 2-4 でのリスク**: CatBoost が 9h タイムアウトを引き起こす可能性あり
> → LB が出ない場合は CB を削除した版を試す

---

## 参考: 上位解 (LB 9.934) の構成

```python
# LightGBM × 3 seeds + CatBoost × 1
# Ridge stacking (positive weights)
# → rogii Phase 2-4 と同じ構成を採用済み
```

競合もほぼ同じ構成 → **モデル選択より特徴量が重要**

Formation 特徴量 (ANCC/ASTNU 等) が本質的な差分

---

*作成: 2026-06-21*
*rogii-wellbore-geology Kaggle competition*
