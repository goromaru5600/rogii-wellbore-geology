# 銀メダル獲得ロードマップ
## ROGII Wellbore Geology Prediction

> 作成日: 2026-06-17  
> データソース: Kaggle API（リーダーボード上位100件 + 公開ノートブック上位20件）  
> コンペ締切: 2026-08-05

---

## 現在のリーダーボード状況（2026-06-17）

| 順位 | スコア（評価値） | 状況 |
|------|----------------|------|
| 1位  | **5.670** | トップ |
| 5位  | 6.429 | |
| 10位 | 6.658 | |
| 20位 | 6.983 | |
| 50位 | 7.332 | |
| 100位 | 7.506 | |

参加チーム数: **3,130チーム**（最終締切まで増加見込み）

---

## メダルボーダー試算

| メダル | Kaggle基準 | 推定必要順位 | 目標スコア |
|--------|-----------|------------|----------|
| 🥇 Gold | top 0.5% or top 10 | 上位16位以内 | **< 7.0** |
| 🥈 Silver | top 2% or top 25 | **上位63位以内** | **< 7.4** |
| 🥉 Bronze | top 10% or top 50 | 上位313位以内 | < 8.x（推定） |

> 100位スコアが7.506のため、銀メダルボーダー（63位相当）は**7.3〜7.4**と推定

---

## 上位公開ノートブック（攻略ヒント）

| 投票数 | タイトル | 著者 | キーテクニック |
|--------|--------|------|--------------|
| 603 | 9.251 DWT-based | parthenos | DWT（離散ウェーブレット変換）によるGRマッチング |
| 241 | Ridge SP Pipeline | Mahdi Ravaghi | Ridge回帰 + 信号処理パイプライン |
| 210 | Dual Pipeline Blend | PyJa | 2つの独立パイプラインをブレンド |
| 194 | Hill Climbing | Mahdi Ravaghi | アライメントのヒルクライミング最適化 |
| 188 | Dual Pipeline + Self-Verifying | Vladislav Lavrentev | 自己検証付きデュアルパイプライン |
| 184 | Target-Free Geosteering | Pilkwang Kim | テストデータのみで完結する地層アライメント |
| 172 | SUPER SOLUTION \| TOP 3 | Roman Tamrazov | （LB TOP 3 達成手法 - 要フォーク） |
| 166 | XGB Starter [CV 15] | Chris Deotte | XGBoost ベースライン |
| 164 | h-blend v1 | f.a.nina | ブレンドアンサンブル |
| 153 | EDA + Target-Free Alignment | Pilkwang Kim | EDA + アライメント詳解 |

### 最重要テクニックまとめ

| 技術 | 概要 | 難易度 | 効果 |
|------|------|--------|------|
| **GR パターンマッチング（DWT/NCC）** | 水平井のGRをtypewellに照合し地層境界を推定 | 中 | 高 |
| **デュアルパイプライン** | 信号処理系モデル + 機械学習モデルの並列構築 | 高 | 高 |
| **ヒルクライミング最適化** | TVT_input を微調整して誤差を最小化 | 中 | 中 |
| **Ridge/XGB ブレンド** | 複数モデルの予測を混合 | 低 | 中 |
| **Self-Verifying アライメント** | アライメント結果の整合性を自己検証 | 高 | 高 |

---

## フェーズ別ロードマップ

### Phase 1: ベースライン確立（〜2026-06-22）

**目標スコア: < 12.0（まずLBに乗る）**

- [ ] `TVT_input` をそのまま提出 → ベーススコア確認
- [ ] Chris Deotte の XGB Starter（CV 15相当）をフォーク・実行
- [ ] Mahdi Ravaghi の LightGBM NB をフォーク・実行
- [ ] Roman Tamrazov の Super Baseline (LB: 12.602) の構造を理解
- [ ] **最初のLB提出を完了させる**

```
スコア目安: TVT_input提出 → ~30, LightGBM → ~10〜12
```

---

### Phase 2: GRパターンマッチングの実装（〜2026-07-06）

**目標スコア: < 9.0**

#### 2-1. DWT ベースのGRアライメント

```python
# typewell の GR シグナルと horizontal_well の GR シグナルを
# DWT（離散ウェーブレット変換）で照合し、TVT 境界を推定

from scipy.signal import correlate
import pywt

# 1. typewell の GR を多解像度分解
# 2. horizontal_well の GR を typewell 空間にマッピング
# 3. マッチング位置からオフセットを計算
# 4. 地層境界（ANCC等）の擬似特徴量を生成
```

- [ ] DWT-based NB (603票) の手法を理解・再実装
- [ ] NCC（正規化相互相関）によるGR照合を実装
- [ ] 最近傍 well（座標ベース）の特定・活用
- [ ] 擬似地層境界特徴量を LightGBM に追加

#### 2-2. 特徴量エンジニアリング

| 特徴量グループ | 内容 |
|--------------|------|
| **GR統計量** | rolling mean/std（窓幅: 10/50/200ft） |
| **TVT_input派生** | diff, cumsum, 局所傾き |
| **typewell参照** | アライメント後のtypewell GR・Geology |
| **座標特徴量** | 近傍 well との距離・角度 |
| **深度特徴量** | MD正規化、セクション内相対位置 |

```
スコア目安: DWT + LightGBM → ~8.5〜9.0
```

---

### Phase 3: デュアルパイプライン構築（〜2026-07-20）

**目標スコア: < 7.8**

#### 3-1. パイプライン A: 信号処理系（TVT_input 補正）

```
TVT_input + GR パターンマッチング
    → オフセット補正モデル（typewell基準）
    → 境界推定 + 内挿補間
    → Prediction A
```

- [ ] typewell GR との depth alignment を精緻化
- [ ] 地層境界の確率的推定（softmax形式）
- [ ] Ridge回帰による補正項の推定（Mahdi Ravaghi の手法参照）

#### 3-2. パイプライン B: 機械学習系

```
全特徴量（GR + 座標 + typewell参照 + Phase2特徴量）
    → XGBoost / LightGBM（well-level LOO CV）
    → Prediction B
```

- [ ] Well-level Leave-One-Out CV の実装
- [ ] Optuna によるハイパーパラメータ最適化
- [ ] OOF スコアと LB スコアの相関確認

#### 3-3. パイプライン A + B のブレンド

- [ ] 単純加重平均（初期: 50:50）
- [ ] Well ごとにどちらが強いかを分析
- [ ] 加重の最適化

```
スコア目安: デュアルパイプライン → ~7.5〜8.0
```

---

### Phase 4: 高度化 & アンサンブル（〜2026-08-03）

**目標スコア: < 7.4（銀メダル圏）**

#### 4-1. ヒルクライミングによるアライメント最適化

```python
# アライメントパラメータ（オフセット・スケール）を
# LB提出スコアではなくCV スコアを目的関数として最適化

def hill_climbing_alignment(well_data, typewell_data, n_iter=1000):
    best_offset = 0
    best_score = float('inf')
    for _ in range(n_iter):
        candidate_offset = best_offset + np.random.randn() * step_size
        score = evaluate_alignment(well_data, typewell_data, candidate_offset)
        if score < best_score:
            best_score = score
            best_offset = candidate_offset
    return best_offset
```

- [ ] Mahdi Ravaghi の Hill Climbing NB を参照・改良
- [ ] Self-Verifying: アライメント信頼度スコアを計算・低信頼度 well を特定
- [ ] 低信頼度 well に対して代替手法（物理モデル）を適用

#### 4-2. ニューラルネットワーク追加

- [ ] 1D CNN または TCN（Temporal Convolutional Network）で GR 系列をモデリング
- [ ] LSTM/GRU による深度方向の連続性学習
- [ ] NN予測を既存パイプラインに追加（3モデルブレンド）

#### 4-3. 最終アンサンブル

- [ ] Pipeline A + Pipeline B + NN の3モデルブレンド
- [ ] Optuna で各モデルの重みを最適化（CV ベース）
- [ ] 提出戦略: 最終5日間で最大5提出/日 × 5日 = 最大25提出で調整

```
スコア目安: 最終アンサンブル → 7.3〜7.4
```

---

## CVとLBの関係（重要）

このコンペはテスト well が **3本**と極めて少ないため、LB スコアが安定しにくい。

| 課題 | 対策 |
|------|------|
| LBのノイズが大きい | Well-level LOO CVを信頼し、LBに過度に依存しない |
| Well間の分布が異なる | CV時も各well単位でスコアを確認 |
| Overfitting リスク | LBより高いCVスコアが出たら過学習を疑う |

---

## 週次チェックポイント

| 週 | 期間 | マイルストーン | 目標LBスコア |
|----|------|--------------|------------|
| Week 1 | 6/17 - 6/22 | ベースライン提出・EDA完了 | < 12.0 |
| Week 2 | 6/23 - 6/29 | DWT GR マッチング実装 | < 9.5 |
| Week 3 | 6/30 - 7/06 | GR特徴量 + LightGBM改善 | < 9.0 |
| Week 4 | 7/07 - 7/13 | デュアルパイプライン A 構築 | < 8.5 |
| Week 5 | 7/14 - 7/20 | パイプライン B + ブレンド | < 8.0 |
| Week 6 | 7/21 - 7/27 | ヒルクライミング + NN追加 | < 7.6 |
| Week 7 | 7/28 - 8/03 | 最終アンサンブル・調整 | **< 7.4** |
| Final  | 8/04 - 8/05 | 最終2提出の選択 | ✅ 銀メダル |

---

## 即座に着手すべきアクション（今週）

1. **`TVT_input` をそのまま Kaggle Notebook で Submit** してベーススコアを確認
2. **DWT-based NB（603票）をフォーク**して Kaggle 上で実行・スコア確認
3. **Roman Tamrazov の SUPER SOLUTION（TOP 3 達成）をフォーク** → 最先端のアプローチを把握
4. **EDA NB（Pilkwang Kim, 153票）を読む** → typewell 〜 horizontal well の関係を視覚的に理解
5. **Chris Deotte の XGB Starter（166票）をフォーク** → tabular ML の出発点を確保

---

## 参考リンク（Kaggle NB）

| NB | URL |
|----|-----|
| DWT-based（9.251） | `https://www.kaggle.com/code/nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based` |
| SUPER SOLUTION TOP 3 | `https://www.kaggle.com/code/romantamrazov/rogii-super-solution-lb` |
| Ridge SP Pipeline | `https://www.kaggle.com/code/maraviaghi/wellbore-geology-prediction-ridge` |
| Hill Climbing | `https://www.kaggle.com/code/maraviaghi/wellbore-geology-prediction-hill-climbing` |
| Dual Pipeline + Self-Verifying | `https://www.kaggle.com/code/vladislavlavrentev/rogii-dual-pipeline-self-verifying` |
| Target-Free Geosteering | `https://www.kaggle.com/code/pilkwang/rogii-target-free-tvt-geosteering` |
| XGB Starter [CV 15] | `https://www.kaggle.com/code/cdeotte/xgb-starter-cv-15` |
| EDA + Target-Free Alignment | `https://www.kaggle.com/code/pilkwang/rogii-eda-target-free-alignment-for-tvt` |

---

*このファイルはKaggle API（リーダーボード上位100件 + 公開NB投票数データ）をもとに作成*
