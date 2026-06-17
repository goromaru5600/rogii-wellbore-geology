# スコア 9 台達成のための技術調査
## ROGII Wellbore Geology Prediction

> 作成日: 2026-06-17  
> データソース: 上位7ノートブックのソースコード解析（Kaggle API でダウンロード）  
> 現在のスコア: 28.341 → 目標: < 9.5

---

## 解析したノートブック

| 投票数 | タイトル | LBスコア | 主要手法 |
|--------|--------|---------|---------|
| 603 | DWT-based | **9.251** | LightGBM+CatBoost + DWT + NCC + 粒子フィルター |
| 241 | Ridge SP Pipeline | 不明 | NNLS ブレンド + PF ポスト処理 |
| 186 | Target-Free Geosteering | **8.905** | HGB+XGB+CB NNLS ブレンド（184特徴量） |
| 167 | XGB Starter | 不明 | drift ターゲット + typewell GR 特徴量 |
| 147 | LightGBM | 不明 | Optuna チューニング + 163特徴量 |
| 109 | Physics-Informed | **8.905** | 128シード粒子フィルター + ビームサーチ |
| 108 | NCC Drift Targeting | 不明 | マルチスケール NCC + Formation KNN |

---

## 最重要発見: ターゲット変換

**現在のベースラインの根本的な問題**: TVT の絶対値（11,000〜12,000 ft）を予測しようとしている。

全ての上位ノートブックが共通して実装している変換:

```python
# ❌ 現在のベースライン（スケールが大きすぎて誤差が増幅）
pred = last_tvt + slope * (MD - last_MD)  # LB: 28.341

# ✅ 上位手法（drift = TVT のずれ量を予測）
anchor_tvt = last_known_TVT_input  # アンカー点（NaN 直前の最後の既知値）
drift_target = TVT - anchor_tvt    # ±16 ft 程度の小さな残差
pred = anchor_tvt + model.predict(drift_features)
```

**効果**: この変換だけで 19.5 → 14.99 ft RMSE（XGB Starter の報告）

---

## スコア改善のための技術要素（優先度順）

### 1. drift ターゲット変換 ★★★（必須・即効性大）

```python
# データ分割
anchor = hw[hw['TVT_input'].notna()]   # アンカー区間（TVT_input が非 NaN）
eval_zone = hw[hw['TVT_input'].isna()] # 予測区間（提出が必要な行）

# ターゲット
last_anchor_tvt = anchor['TVT_input'].iloc[-1]
drift = eval_zone['TVT'] - last_anchor_tvt  # 学習ターゲット
```

### 2. GroupKFold by well_id ★★★（CV の信頼性に必須）

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for trn_idx, val_idx in gkf.split(X, y, groups=well_ids):
    # well 単位で分割 → 同じ well が train/val に混在しない
    model.fit(X.iloc[trn_idx], y.iloc[trn_idx])
```

> ⚠️ 通常の KFold を使うと同じ well の行が train/val 両方に入り、CV スコアが楽観的になる

### 3. Typewell GR 特徴量 ★★★（NCC の簡易版）

```python
# typewell の GR を参照点として水平井 GR との差分を特徴量化
tw_gr_at_baseline = np.interp(baseline_tvt, typewell['TVT'], typewell['GR'])
gr_diff = horizontal_well['GR'] - tw_gr_at_baseline  # typewell との GR 差分
```

### 4. マルチスケール NCC（GR バーコードマッチング） ★★★（9 台の核心）

typewell の GR パターンと水平井 GR の局所相関で TVT を推定する。

```python
def multi_scale_ncc(typewell_gr, typewell_tvt, hw_gr, half_widths=(8, 15, 25)):
    """
    水平井の GR 窓と typewell GR の正規化相互相関を計算し、
    最も一致する typewell TVT を返す。
    """
    results = []
    for hw in half_widths:
        win = 2 * hw + 1
        scores = []
        for i, candidate_tvt_idx in enumerate(range(len(typewell_tvt))):
            # typewell の GR 窓
            tw_win = typewell_gr[max(0, candidate_tvt_idx-hw):candidate_tvt_idx+hw+1]
            # 水平井の GR 窓
            hw_win = hw_gr[-win:]
            if len(tw_win) == win and len(hw_win) == win:
                # ピアソン相関（振幅不変）
                score = np.corrcoef(tw_win, hw_win)[0, 1]
                scores.append((score, typewell_tvt[candidate_tvt_idx]))
        best_tvt = max(scores, key=lambda x: x[0])[1]
        results.append((best_tvt, max(s[0] for s in scores)))

    # スコア重み付きアンサンブル（softmax）
    tvts = np.array([r[0] for r in results])
    scores = np.array([r[1] for r in results])
    weights = np.exp(3 * scores); weights /= weights.sum()
    return (tvts * weights).sum()
```

### 5. Formation KNN（近傍 well の地層深度を空間補間） ★★

訓練 well の地層境界（ANCC 等）を座標から KNN + 平面フィットで補間。

```python
from sklearn.neighbors import NearestNeighbors

# 訓練 well の ANCC 深度を記録
train_formation = []
for well_id, df in train_wells:
    x, y = df['X'].mean(), df['Y'].mean()
    ancc_depth = df['ANCC'].mean()
    train_formation.append([x, y, ancc_depth])
train_formation = np.array(train_formation)

# テスト well の座標から k=10 最近傍 well を検索
knn = NearestNeighbors(n_neighbors=10)
knn.fit(train_formation[:, :2])
distances, indices = knn.kneighbors([[test_X, test_Y]])

# 2D 平面フィット（地層傾斜を考慮）
neighbors = train_formation[indices[0]]
plane_fit = np.linalg.lstsq(
    np.column_stack([neighbors[:, 0], neighbors[:, 1], np.ones(10)]),
    neighbors[:, 2], rcond=None
)[0]
estimated_ancc = plane_fit[0]*test_X + plane_fit[1]*test_Y + plane_fit[2]

# b_well: アンカー区間で校正
b_well = np.median(anchor['TVT'] + anchor['Z'] - estimated_ancc)
tvt_physics = -eval_zone['Z'] + estimated_ancc + b_well
```

### 6. Savitzky-Golay スムージング（ポスト処理） ★★

```python
from scipy.signal import savgol_filter

# well ごとに予測値をスムージング（w=17, 3次多項式）
for well_id in test_well_ids:
    mask = submission['id'].str.startswith(well_id)
    submission.loc[mask, 'tvt'] = savgol_filter(
        submission.loc[mask, 'tvt'].values,
        window_length=17, polyorder=3
    )
```

### 7. 粒子フィルター（Particle Filter） ★★（8 台への鍵）

GR を観測値として TVT の状態を逐次追跡するベイズフィルター。

```python
def run_particle_filter(eval_gr, typewell_gr, typewell_tvt,
                        n_particles=500, n_seeds=128):
    """
    GR 観測値を使って TVT を逐次推定。
    128 シードの結果を対数尤度で重み付き平均。
    """
    all_preds, all_liks = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        # 初期粒子: アンカー末尾 TVT 周辺に正規分布
        particles = last_anchor_tvt + 2.0 * rng.standard_normal(n_particles)
        weights = np.ones(n_particles) / n_particles
        preds = []
        log_lik = 0.0
        for i, gr_obs in enumerate(eval_gr):
            # 状態遷移（TVT がゆっくり変化）
            particles += rng.standard_normal(n_particles) * 0.5
            # GR 尤度（typewell GR との一致度で重み更新）
            expected_gr = np.interp(particles, typewell_tvt, typewell_gr)
            likelihood = np.exp(-0.5 * ((gr_obs - expected_gr) / 5.0) ** 2)
            weights *= likelihood
            weights /= weights.sum() + 1e-12
            log_lik += np.log(likelihood.mean() + 1e-12)
            preds.append(np.sum(particles * weights))
        all_preds.append(np.array(preds))
        all_liks.append(log_lik)

    # 対数尤度で重み付き平均
    liks = np.array(all_liks)
    w = np.exp(liks - liks.max()); w /= w.sum()
    return (w[:, None] * np.stack(all_preds)).sum(0)
```

---

## 推奨実装順序

### Phase 2-1: drift + typewell GR + XGBoost（目標 < 15）

```python
# 特徴量セット（最小セット）
features = [
    'MD', 'X', 'Y', 'Z', 'GR',           # 基本
    'md_from_anchor',                       # アンカーからの距離
    'gr_rolling_mean_11', 'gr_rolling_std_11',  # GR 統計
    'tw_gr_at_baseline',                    # typewell GR 参照
    'gr_diff_from_tw',                      # typewell との差分
    'slope_last50',                         # アンカー末尾の傾き
    'last_anchor_tvt',                      # アンカー値
]
# ターゲット: drift = TVT - last_anchor_tvt
```

### Phase 2-2: マルチスケール NCC 追加（目標 < 11）

- 窓幅 8 / 15 / 25 の NCC スコアと推定 TVT を特徴量に追加
- 既存の XGBoost モデルに NCC 特徴量を追加するだけで大きく改善

### Phase 2-3: Formation KNN + 粒子フィルター（目標 < 9.5）

- formation_ancc_knn, formation_ancc_b_well を特徴量追加
- 粒子フィルター予測を追加の特徴量または NNLS ブレンド対象に

### Phase 2-4: NNLS アンサンブル（目標 < 9）

- XGBoost + CatBoost + HistGradientBoosting の NNLS ブレンド
- 最終ポスト処理: Savitzky-Golay + オプション: temporal decay

---

## バリデーション設計（重要）

```python
# GroupKFold(5) by well_id で well 単位の汎化性能を測定
gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(X_train))
for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=well_ids)):
    model.fit(X_train.iloc[trn_idx], y_train.iloc[trn_idx])
    oof_preds[val_idx] = model.predict(X_train.iloc[val_idx])

cv_mse = mean_squared_error(y_train, oof_preds)
print(f"OOF MSE (drift): {cv_mse:.4f} → OOF RMSE (drift): {np.sqrt(cv_mse):.4f}")
# これが LB スコアの代理指標になる
```

---

*このファイルは上位7ノートブック（Kaggle API でダウンロード）のソースコード解析をもとに作成*
