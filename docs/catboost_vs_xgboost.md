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
    font-size: 0.82em;
  }
  code {
    font-size: 0.78em;
  }
  h2 {
    color: #2c3e50;
  }
  blockquote {
    border-left: 4px solid #3498db;
    padding-left: 12px;
    color: #555;
  }
---

# CatBoost vs XGBoost vs LightGBM
## 速度・精度・使い分けの判断基準

Kaggle rogii-wellbore-geology コンペ文脈での整理

---

## アルゴリズムの違い（3行で）

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| 木の構築 | Level-wise | **Leaf-wise** | **Symmetric tree** |
| 強み | 安定・実績 | 速度・大規模 | カテゴリ変数・小規模 |
| 勾配計算 | 標準GB | GOSS (高速近似) | **Ordered Boosting** |
| GPU恩恵 | 中 | 中 | **最大** |

> **Leaf-wise**: 最も誤差が大きい葉だけ分割 → 少ない木数で深く学習
> **Symmetric tree**: 全ノードで同じ分割条件 → 予測高速・過学習しにくい
> **Ordered Boosting**: 勾配計算にリーク防止機構 → 小データで強い

---

## 速度比較（CPU / GPU）

```
データ規模: ~380万行 × 57特徴量 (rogii 訓練セット, GroupKFold 5fold)
```

| モデル | iterations | CPU 時間 | GPU 時間 | GPU 倍率 |
|---|---|---|---|---|
| XGBoost | 1,000 | ~5〜10 分 | ~1〜2 分 | 5〜10x |
| LightGBM | 8,000 | ~15〜25 分 | ~5〜8 分 | 3〜5x |
| **CatBoost** | **8,000** | **~30〜60 分** | **~3〜6 分** | **10〜20x** |

**CPU では CatBoost が最も遅い。ただし GPU では最大の恩恵を受ける。**

---

## なぜ CatBoost は CPU で遅いのか

### Ordered Boosting の計算コスト
- 各サンプルの勾配を「時系列順に前のサンプルだけ」で計算
- データを複数のパーミュテーションでシャッフルして繰り返す
- → XGBoost の単純勾配計算より **3〜5倍の演算量**

### Symmetric Tree の反動
- 全ノードで同じ分割 → 深さを増やさないと表現力が出ない
- iterations=8,000 ≈ XGBoost の n_estimators=500〜1,000 相当

### GPU で逆転する理由
- Symmetric tree は GPU の SIMD 並列計算と相性が良い
- 全ノード同時に同じ演算 → GPU が得意なパターン

---

## 使い分けの判断フローチャート

```
データにカテゴリ変数が多い (high cardinality)?
  └─ YES → CatBoost 一択（Target Encoding 不要、リーク防止済み）
  └─ NO  →
        データ行数 > 500万?
          └─ YES → LightGBM（メモリ効率・速度で有利）
          └─ NO  →
                GPU が使える?
                  └─ YES → CatBoost（精度とのトレードオフが最良）
                  └─ NO  → LightGBM（速度と精度のバランス）
```

**XGBoost を選ぶ積極的な理由は現在ほぼない**（LGBMが上位互換に近い）

---

## 精度の使い分け基準

| 状況 | 推奨 | 理由 |
|---|---|---|
| **カテゴリ変数が多い** | CatBoost | 内部Target Encodingが優秀 |
| **数値特徴量のみ** | LightGBM | CB と精度差は小さく速い |
| **小〜中規模データ** (<10万行) | CatBoost | Ordered Boostingでリーク防止 |
| **大規模データ** (>500万行) | LightGBM | メモリ・速度で圧倒 |
| **特徴量数が多い** (>500) | LightGBM | GOSS で高速な列サンプリング |
| **アンサンブル目的** | **LGB + CB** | 予測の相関が低く多様性が出る |

---

## アンサンブルで LGB + CB を組み合わせる理由

### 予測の多様性（相関の低さ）が重要

```python
# OOF 予測の相関を確認するのが基本
import numpy as np
corr = np.corrcoef(oof_lgb, oof_cb)[0,1]
# 相関が 0.95 未満なら組み合わせる価値あり
# → LGB と CB は内部アルゴリズムが全く異なるので相関が下がりやすい
```

| ペア | 典型的相関 | アンサンブル効果 |
|---|---|---|
| LGB seed1 vs seed2 | ~0.99 | 小さい |
| LGB vs XGBoost | ~0.97 | 小さい |
| **LGB vs CatBoost** | **~0.90〜0.95** | **大きい** |

---

## Kaggle での実践的な使い分け

### CPU 環境（無料枠・デフォルト）
```
XGBoost  → 素早く動かしてベースラインを作る
LightGBM → メインモデル（速度・精度のバランス最良）
CatBoost → iterations を 3,000〜5,000 に抑えて追加
```

### GPU 環境（T4 / P100）
```
LightGBM → CPU のまま or device='gpu' で高速化
CatBoost → task_type='GPU' で 10〜20x 高速化 → メインに昇格可能
```

> **結論**: GPU があれば CatBoost を積極的に使う。CPU のみなら LightGBM 中心で CB は iterations 少なめに。

---

## パラメータ対応表（移植時の参考）

| 概念 | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| 木の数 | `n_estimators` | `n_estimators` | `iterations` |
| 学習率 | `learning_rate` | `learning_rate` | `learning_rate` |
| 木の深さ | `max_depth` | `max_depth` / `num_leaves` | `depth` |
| 正則化 (L2) | `reg_lambda` | `reg_lambda` | `l2_leaf_reg` |
| サブサンプル | `subsample` | `subsample` | `subsample` |
| 早期終了 | `early_stopping_rounds` | `callbacks=[early_stopping()]` | `od_wait` |
| GPU 有効化 | `device='cuda'` | `device='gpu'` | `task_type='GPU'` |

---

## rogii Phase 2-4 での構成と判断根拠

```python
# LightGBM × 3 seeds → メイン（速度・安定性）
LGB_CONFIGS = [
    dict(learning_rate=0.025, n_estimators=8000, seed=42),
    dict(learning_rate=0.020, n_estimators=8000, seed=7),
    dict(learning_rate=0.030, n_estimators=8000, seed=123),
]
# CatBoost × 1 + GPU → 多様性確保（CB は内部アルゴリズムが異なる）
CB_PARAMS = dict(iterations=8000, depth=7, task_type='GPU', od_wait=300)
# Ridge Stacking → 重みを OOF で最適化（正値制約付き）
ridge = Ridge(alpha=1., positive=True)
```

**XGBoost を外した理由**: LightGBM と相関が高い割に速度が遅い → 多様性への貢献が薄い

---

## まとめ：使い分けチートシート

| 判断軸 | 選択 |
|---|---|
| カテゴリ変数が主役 | **CatBoost** |
| 数値特徴量のみ・大規模 | **LightGBM** |
| GPU あり | **CatBoost** を積極採用 |
| CPU のみ・時間制限あり | **LightGBM** メイン、CB は iterations 減 |
| アンサンブル | **LGB × 複数seed + CB × 1** が鉄板 |
| XGBoost を選ぶ場面 | 既存コードの流用 / 互換性重視のみ |

> **今回の rogii**: 数値特徴量のみ → LGB メイン、GPU 有効で CB 追加、XGB 不使用

---

*作成: 2026-06-21*
*rogii-wellbore-geology Kaggle competition*
