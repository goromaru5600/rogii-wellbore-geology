---
marp: true
theme: default
paginate: true
style: |
  section { font-size: 21px; font-family: 'Helvetica Neue', sans-serif; }
  h1 { color: #1a4f7a; border-bottom: 2px solid #1a4f7a; }
  h2 { color: #2c7ab3; }
  table { font-size: 17px; }
  code { background: #f0f4f8; font-size: 16px; }
  .critical { background: #ffeeba; padding: 6px 10px; border-left: 4px solid #e0a800; }
  .insight { background: #d4edda; padding: 6px 10px; border-left: 4px solid #28a745; }
---

# 事前に知っておくべき重要知識
## ROGII Wellbore Geology Prediction

---

# 目次

1. 油ガス業界のドメイン知識
2. 深度の3種類（MD / TVD / TVT）
3. GRログとは何か
4. データ構造の本質（最重要）
5. タスクの本質：何を予測すべきか
6. 特徴量エンジニアリング戦略
7. 時系列・信号処理の手法
8. モデリング戦略
9. クロスバリデーション設計
10. コンペ固有の注意事項

---

# 1. 油ガス業界のドメイン知識

## 水平井掘削とは

```
地表
 │  ← 垂直部分（まず縦に掘る）
 │
 ╰──────────────────────────────→  水平部分（横に延びる "Lateral"）
                                    └ ここで石油・ガスを採掘
```

- **ターゲット地層**（例: Eagle Ford シェール）に水平に入り込み採掘
- 水平部分は数千フィートに及ぶ
- **ジオステアリング**: 地質家がリアルタイムで「今どの地層にいるか」を判断し掘削方向を制御
- 地層から外れると採掘効率が激減 → 正確な位置把握が重要

## 本コンペの文脈

> 「ジオステアリングの自動化」= 機械学習でリアルタイム地層位置予測

---

# 2. 深度の3種類（MD / TVD / TVT）

| 指標 | 正式名称 | 意味 | 本コンペでの役割 |
|------|---------|------|----------------|
| **MD** | Measured Depth | 掘削管の実際の長さ（沿坑距離） | 主要インデックス |
| **TVD** | True Vertical Depth | 鉛直方向の深さ（Z カラム = -TVD） | 特徴量 |
| **TVT** | True Vertical Thickness | **地層内の垂直位置** | 🎯 予測対象 |

## TVT ≠ TVD の理由

地層は水平ではなく**傾いている（地層傾斜）**ため、
鉛直深度だけでは「地層内のどこにいるか」がわからない

```
地表
 │
 ╰─────────────────→  水平井
       ╲ 地層傾斜
        ╲ ANCC 層トップ
         ╲
          ╲ TVT = 傾きを補正した地層内位置
```

---

# 3. GRログとは何か

## Gamma Ray（ガンマ線）ログ

岩石の**自然放射能**を測定するセンサー（API単位）

| GR値 | 地層の種類 | 掘削上の意味 |
|------|-----------|------------|
| **低 (< 50 API)** | 石灰岩・クリーン砂岩 | 炭化水素貯留に有利 |
| **中 (50-100 API)** | シルト岩・泥灰岩 | 中間的 |
| **高 (> 100 API)** | 頁岩・泥岩 | 貯留に不向き |

## なぜ GR が重要か

- **地層の識別**に最も基本的なログ
- テストデータでも唯一利用可能な測定値
- typewell の GR パターンと照合することで**地層内位置を推定**できる

---

# 4. データ構造の本質（最重要）

## 1 well のデータ構造

```
horizontal_well.csv の行:
┌─────────────────────────────────┬──────────────────────────────────────────┐
│ アンカーゾーン（MD: 11467-12908） │ 予測ゾーン（MD: 12909-16744）             │
│ TVT_input = TVT = 正解          │ TVT_input = NaN（初期推定なし）            │
│ 約 27% の行                     │ 約 73% の行 ← ここを予測する！             │
└─────────────────────────────────┴──────────────────────────────────────────┘
         ↑ アンカー = 地質家が         ↑ ここが本来の
           最初に解釈した区間             「わからない部分」
```

## 訓練データとテストデータの違い

| | 訓練データ | テストデータ |
|--|---------|-----------|
| アンカーゾーン | TVT_input = TVT ✅ | TVT_input = TVT ✅（同じ値） |
| 予測ゾーン | **TVT あり（正解）**, TVT_input=NaN | **TVT なし（提出対象）**, TVT_input=NaN |
| ANCC, ASTNU 等 | **全行あり** ✅ | **全行なし** ❌ |

---

# 5. タスクの本質：何を予測すべきか

## 発見した決定的事実

```python
# 相関係数の比較
TVT vs Z           : r = -0.918   ← 普通
TVT vs ANCC        : r =  0.411   ← 低い
TVT vs (Z - ANCC)  : r = -0.9999  ← ほぼ完璧！
```

> **TVT ≈ f(Z - ANCC)**  「ANCCからの相対位置」がTVTをほぼ決定する

## したがって本コンペのコアチャレンジは

```
テスト時に ANCC（地層トップ深度）がない
            ↓
GR ログと typewell（縦井の基準）を使って
ANCC を推定する（= ジオステアリングの本質）
```

→ **GR パターンマッチングで地層内位置を特定する問題**

---

# 6. typewell が鍵：GR-TVT 対応表

## typewell の役割

縦井（typewell）は「地層の縦断面の基準」

```
typewell:
  TVT=11300, GR=82  → ANCC 層（低GR = 石灰岩）
  TVT=11380, GR=130 → ASTNU 層（高GR = 頁岩）
  TVT=11500, GR=72  → EGFDU 層（低GR = 砂岩）
  ...

水平井:
  MD=12500, GR=82 → typewell と GR パターンが一致
                  → この地点は TVT ≈ 11300 付近にいる！
```

## GR ログ照合の手順

1. typewell の `(TVT, GR)` 系列を「基準テンプレート」とする
2. 水平井の GR 系列を typewell にアライメント
3. 一致した TVT 位置 → 水平井の TVT 推定値

---

# 7. 時系列・信号処理の手法

## DTW（Dynamic Time Warping）

GR ログ照合に最も自然なアプローチ

```python
from dtaidistance import dtw
import numpy as np

# typewell の GR テンプレート
template = typewell_df['GR'].values

# 水平井の GR クエリ
query = horizontal_df['GR'].dropna().values

# DTW距離（小さいほど類似）
distance = dtw.distance(query, template)

# パスを取得してTVTをマッピング
_, path = dtw.warping_paths(query, template)
```

**DTW の利点**: 速度の違い（地層の厚み変化）に対してロバスト

---

# 7. 時系列・信号処理の手法（続き）

## Cross-Correlation（相互相関）

```python
from scipy.signal import correlate

# GR シフト量を検出 → TVT のオフセット推定
corr = correlate(horizontal_gr, typewell_gr, mode='full')
lag = np.argmax(corr) - len(typewell_gr) + 1
estimated_tvt_offset = lag * sampling_interval
```

## 特徴量エンジニアリング（GR 系）

```python
# ローリング統計量（地層パターンの局所特性）
df['GR_roll_mean_10'] = df['GR'].rolling(10).mean()
df['GR_roll_std_10']  = df['GR'].rolling(10).std()
df['GR_diff']         = df['GR'].diff()           # 変化率

# typewell との GR 差分（TVTを使ってマッピング）
df['GR_typewell_diff'] = df['GR'] - typewell_at_tvt_input['GR']
```

---

# 8. モデリング戦略

## アプローチ A: 直接予測（シンプル）

```
入力: [MD, X, Y, Z, GR, anchor_context] → 出力: TVT
```
- GBDTやNNで直接TVTを予測
- アンカーゾーンの情報を時系列特徴量として付加

## アプローチ B: 2段階（精度重視）

```
Stage 1: GR + typewell → ANCC の推定
Stage 2: Z - ANCC_estimated → TVT
```
- Z - ANCC が TVT と r=-0.9999 なので Stage 2 は簡単
- Stage 1 の精度がほぼすべてを決める

## アプローチ C: 補正モデル（バランス型）

```
TVT_predicted = TVT_input_estimated + ΔCorrection
```
- アンカーゾーンの勾配から初期TVT外挿 → GRで補正
- 上位解法の多くがこのパターンを採用する可能性

---

# 9. 特徴量エンジニアリングの全体像

```python
# アンカーゾーン由来の特徴量（コンテキスト）
df['anchor_last_tvt']       = anchor_df['TVT'].iloc[-1]   # 最後の既知TVT
df['anchor_tvt_slope']      = np.polyfit(anchor_df['MD'], anchor_df['TVT'], 1)[0]
df['md_from_anchor_end']    = df['MD'] - anchor_df['MD'].max()

# 座標系特徴量
df['md_increment']          = df['MD'].diff()
df['z_gradient']            = df['Z'].diff() / df['MD'].diff()

# typewell 照合特徴量（GRによる地層位置推定）
df['estimated_tvt_from_gr'] = dtw_map(df['GR'], typewell)
df['gr_vs_typewell_diff']   = df['GR'] - typewell_at(df['estimated_tvt_from_gr'])

# GR ローリング統計
for w in [5, 10, 20, 50]:
    df[f'GR_mean_{w}'] = df['GR'].rolling(w).mean()
    df[f'GR_std_{w}']  = df['GR'].rolling(w).std()
```

---

# 10. クロスバリデーション設計

## 絶対にやってはいけない：ランダムCV

```python
# ❌ これは絶対NG
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)
```

同一 well 内の行は**強い自己相関**がある → データリークが発生

## 正しいアプローチ：Well-Level CV (LOGO CV)

```python
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()

# groups = well_id ごとのラベル
for train_idx, val_idx in logo.split(X, y, groups=well_ids):
    ...
```

**LOGO CV（Leave-One-Group-Out）**: 1 wellを丸ごとバリデーションにする

→ テスト時の状況（未知のwellに対して予測）を正確に再現

---

# 11. MSE 評価指標の性質

## MSE vs RMSE

```
評価指標: MSE（Mean Squared Error）= (1/N) Σ(TVT - TVT_hat)²

例: 2つの予測
  予測A: 全行が 2ft ずれ    → MSE = 4.0
  予測B: 1行だけ 10ft ずれ  → MSE = (1/N) × 100 ≫ 予測A
```

> **外れ値が MSE を支配する** → アウトライアーの処理が重要

## 実務的な対策

```python
# 予測値のクリッピング（物理的に妥当な範囲に制約）
pred = np.clip(pred, tvt_min - 50, tvt_max + 50)

# アンサンブルで外れ値を平滑化
final_pred = 0.5 * lgbm_pred + 0.3 * xgb_pred + 0.2 * baseline_pred
```

---

# 12. コンペ固有の注意事項

## Kernel Only（Notebook提出必須）

```
ローカル学習 → モデルを Kaggle Dataset としてアップロード
                        ↓
          Kaggle Notebook で推論 → submission.csv を出力
```

```python
# モデルの保存（ローカル）
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
# → Kaggle Dataset にアップロード

# Notebook での読み込み
with open('/kaggle/input/my-rogii-model/model.pkl', 'rb') as f:
    model = pickle.load(f)
```

## その他の制約

| 制約 | 内容 |
|------|------|
| 提出回数 | **5回/日** まで |
| テスト well 数 | **3 well のみ** → LB スコアの分散が大きい |
| 推論時間 | Kaggle Notebook のタイムアウトに注意 |

---

# 13. テストwell 3本問題

## なぜこれが重要か

- **3 well だけで評価** → Public LB はノイズが大きい
- 1 well での大きな外れ値が順位を大きく変える可能性

## 対策

```
1. Local CV（LOGO）を信頼する
   → Public LB に過度に最適化しない

2. CVとLBのギャップを記録する
   → 乖離が大きい場合はデータリークを疑う

3. 安定した予測を優先
   → 予測値の分布が訓練データのTVT分布と近いか確認
```

## 銅メダルの条件

- **上位 10%（約 227 チーム以内）**
- 現在 100 位スコア ≈ **8.49 (MSE 換算)**
- ベースラインを超えるだけで圏内に入れる可能性あり

---

# 14. 公開ノートブックから学ぶ戦略

| ノートブック | スコア（RMSE相当） | キーテクニック |
|-------------|------------------|---------------|
| Super Baseline | 12.602 | 基本的なGBDT |
| LightGBM Baseline | - | LightGBM + 基本特徴量 |
| Better Solution | 9.956 | 改良版特徴量 |
| DWT-based | **9.251** | 離散ウェーブレット変換 |
| 現在1位 | **5.986** | 非公開 |

> **9.251 → 5.986 のギャップ** = 上位解法の差別化要因
> → typewell 活用の深度、時系列モデル、アンカー活用が鍵

---

# 15. 推奨ライブラリ

```python
# 時系列・信号処理
pip install dtaidistance   # DTW
pip install tslearn         # 時系列ML（DTW + clustering）
from scipy.signal import correlate  # 相互相関

# 特徴量エンジニアリング
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d  # TVT補間

# モデリング
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# 可視化（well ログ）
import matplotlib.pyplot as plt
# well ログは深度をY軸（下方向）で描く慣習
fig, ax = plt.subplots()
ax.plot(df['GR'], df['TVT'])
ax.invert_yaxis()  # 深くなるほど下
```

---

# まとめ：攻略の核心

```
1. Z - ANCC が TVT とほぼ完全相関（r=-0.9999）
   → ANCC 推定精度が最終スコアを決める

2. ANCC の推定 = typewell GR と水平井 GR のパターンマッチング
   → DTW / 相互相関 が基礎手法

3. アンカーゾーン（最初の 27% の行）を最大活用
   → 既知のTVT＋GRパターンから予測ゾーンへ外挿

4. Well-Level CV（LOGO）で正しく評価
   → ランダムCVはデータリークで過大評価になる

5. MSE は外れ値に敏感
   → クリッピング・アンサンブルでロバスト性を確保
```

> **DWT ベースが LB 9.251 を達成済み → DTW/相関ベースで 9 切りを狙う**
