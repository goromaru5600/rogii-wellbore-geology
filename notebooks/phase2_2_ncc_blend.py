"""
Phase 2-2: NCC GR Alignment + XGBoost Blend
目標スコア: < 10.0 (現在: 14.268)

アプローチ:
- Pipeline A: NCC（正規化相互相関）でtypewell GRと水平井GRをアライメント → TVT直接推定
- Pipeline B: Phase2-1 と同じ Drift XGBoost（NCC特徴量を追加）
- 最終予測 = NCC_ALPHA * Pipeline_A + (1-NCC_ALPHA) * Pipeline_B
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ── データパス設定 ───────────────────────────────────────────────
CANDIDATES = [
    '/kaggle/input/rogii-wellbore-geology-prediction',
    '/kaggle/input/competitions/rogii-wellbore-geology-prediction',
]
DATA_DIR = None
for c in CANDIDATES:
    if os.path.exists(os.path.join(c, 'train')):
        DATA_DIR = c
        break

if DATA_DIR is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(_here, '..', 'data')

TRAIN_DIR  = os.path.join(DATA_DIR, 'train')
TEST_DIR   = os.path.join(DATA_DIR, 'test')
SAMPLE_SUB = os.path.join(DATA_DIR, 'sample_submission.csv')
print(f"DATA_DIR: {DATA_DIR}")

NCC_ALPHA = 0.6   # NCC の重み（0=XGBのみ, 1=NCCのみ）


# ── ユーティリティ ───────────────────────────────────────────────
def load_well(well_id, base_dir):
    hw = pd.read_csv(os.path.join(base_dir, f'{well_id}__horizontal_well.csv'))
    tw = pd.read_csv(os.path.join(base_dir, f'{well_id}__typewell.csv'))
    hw['GR'] = hw['GR'].interpolate(limit_direction='both')
    hw['GR'] = hw['GR'].fillna(hw['GR'].median())
    return hw, tw


def smooth_gr(gr, window=11):
    if len(gr) < window:
        return gr
    return savgol_filter(gr, window_length=window, polyorder=3)


# ── Pipeline A: NCC GR アライメント ─────────────────────────────
def ncc_align(anchor, tw, search_range=80, step=0.5):
    """
    アンカー区間の GR を typewell GR と照合し、ベスト TVT オフセットを返す。
    Returns: best_offset [ft], best_ncc_score [-1〜1]
    """
    tw_interp = interp1d(
        tw['TVT'].values, tw['GR'].values,
        bounds_error=False, fill_value='extrapolate'
    )
    tail = anchor.tail(300)
    if len(tail) < 20:
        return 0.0, 0.0

    anchor_tvt = tail['TVT_input'].values
    anchor_gr  = smooth_gr(tail['GR'].values)
    offsets    = np.arange(-search_range, search_range + step, step)
    scores     = []

    for offset in offsets:
        tw_gr = tw_interp(anchor_tvt + offset)
        valid = np.isfinite(tw_gr) & np.isfinite(anchor_gr)
        if valid.sum() < 10:
            scores.append(-999.0)
            continue
        ag, tg = anchor_gr[valid], tw_gr[valid]
        if ag.std() < 1e-6 or tg.std() < 1e-6:
            scores.append(0.0)
            continue
        score = np.corrcoef(ag, tg)[0, 1]
        scores.append(float(score) if np.isfinite(score) else -999.0)

    best_idx = int(np.argmax(scores))
    return float(offsets[best_idx]), float(scores[best_idx])


def predict_ncc(hw, tw, anchor, eval_zone):
    """NCC アライメントで eval_zone の TVT を推定（絶対値）。"""
    ncc_offset, ncc_score = ncc_align(anchor, tw)

    last_anchor_tvt = anchor.iloc[-1]['TVT_input']
    last_anchor_md  = anchor.iloc[-1]['MD']

    tail = anchor.tail(100)
    slope = np.polyfit(tail['MD'], tail['TVT_input'], 1)[0] if len(tail) >= 5 else 0.0
    slope = np.clip(slope, -0.02, 0.02)

    md_from_anchor = eval_zone['MD'].values - last_anchor_md
    baseline_tvt   = last_anchor_tvt + slope * md_from_anchor
    ncc_tvt        = baseline_tvt + ncc_offset

    if len(ncc_tvt) >= 17:
        ncc_tvt = savgol_filter(ncc_tvt, window_length=17, polyorder=3)

    return ncc_tvt, ncc_offset, ncc_score


# ── Pipeline B: 特徴量構築 ──────────────────────────────────────
def build_features(hw, tw, anchor, eval_zone, ncc_offset=0.0, ncc_score=0.0):
    n = len(eval_zone)
    if n == 0:
        return pd.DataFrame()

    tw_interp = interp1d(
        tw['TVT'].values, tw['GR'].values,
        bounds_error=False, fill_value=(tw['GR'].values[0], tw['GR'].values[-1])
    )

    last_anchor     = anchor.iloc[-1]
    last_anchor_tvt = last_anchor['TVT_input']
    last_anchor_md  = last_anchor['MD']
    last_anchor_z   = last_anchor['Z']

    tail = anchor.tail(100)
    slope_tvt = np.polyfit(tail['MD'], tail['TVT_input'], 1)[0] if len(tail) >= 5 else 0.0

    anchor_gr_mean = anchor['GR'].tail(50).mean()
    anchor_gr_std  = anchor['GR'].tail(50).std() + 1e-6

    md_arr = eval_zone['MD'].values
    z_arr  = eval_zone['Z'].values
    gr_arr = eval_zone['GR'].values

    md_from_anchor = md_arr - last_anchor_md
    z_from_anchor  = z_arr  - last_anchor_z

    slope_clipped = np.clip(slope_tvt, -0.02, 0.02)
    baseline_tvt  = last_anchor_tvt + slope_clipped * md_from_anchor

    tw_gr_at_base    = tw_interp(baseline_tvt)
    tw_gr_at_ncc     = tw_interp(baseline_tvt + ncc_offset)
    tw_gr_at_base_m5 = tw_interp(baseline_tvt - 5)
    tw_gr_at_base_p5 = tw_interp(baseline_tvt + 5)
    tw_gr_at_base_m10 = tw_interp(baseline_tvt - 10)
    tw_gr_at_base_p10 = tw_interp(baseline_tvt + 10)

    full_gr  = pd.Series(pd.concat([anchor['GR'], eval_zone['GR']]).values)
    anc_len  = len(anchor)
    row_frac = np.linspace(0, 1, n)

    feats = pd.DataFrame({
        'md_from_anchor'    : md_from_anchor,
        'z_from_anchor'     : z_from_anchor,
        'row_frac'          : row_frac,
        'Z'                 : z_arr,
        'GR'                : gr_arr,
        'gr_norm'           : (gr_arr - anchor_gr_mean) / anchor_gr_std,
        'gr_rm11'           : full_gr.rolling(11,  min_periods=1, center=True).mean().values[anc_len:],
        'gr_rm51'           : full_gr.rolling(51,  min_periods=1, center=True).mean().values[anc_len:],
        'gr_rm151'          : full_gr.rolling(151, min_periods=1, center=True).mean().values[anc_len:],
        'gr_rs11'           : full_gr.rolling(11,  min_periods=1, center=True).std().fillna(0).values[anc_len:],
        'gr_rs51'           : full_gr.rolling(51,  min_periods=1, center=True).std().fillna(0).values[anc_len:],
        'gr_diff_tw'        : gr_arr - tw_gr_at_base,
        'gr_diff_tw_ncc'    : gr_arr - tw_gr_at_ncc,
        'gr_diff_tw_m5'     : gr_arr - tw_gr_at_base_m5,
        'gr_diff_tw_p5'     : gr_arr - tw_gr_at_base_p5,
        'tw_gr_at_base'     : tw_gr_at_base,
        'tw_gr_at_ncc'      : tw_gr_at_ncc,
        'tw_gr_at_base_m10' : tw_gr_at_base_m10,
        'tw_gr_at_base_p10' : tw_gr_at_base_p10,
        'last_anchor_tvt'   : last_anchor_tvt,
        'last_anchor_z'     : last_anchor_z,
        'slope_tvt'         : slope_tvt,
        'anchor_gr_mean'    : anchor_gr_mean,
        'anchor_gr_std'     : anchor_gr_std,
        'ncc_offset'        : ncc_offset,
        'ncc_score'         : ncc_score,
        'X'                 : eval_zone['X'].values,
        'Y'                 : eval_zone['Y'].values,
    })
    return feats


# ── 学習データ構築 ───────────────────────────────────────────────
print("Loading train data...")
train_files  = sorted(glob.glob(os.path.join(TRAIN_DIR, '*__horizontal_well.csv')))
well_ids_all = [os.path.basename(f).replace('__horizontal_well.csv', '') for f in train_files]

all_X, all_y, all_groups = [], [], []
all_ncc_tvt   = []   # ブレンド OOF 評価用: 各行の NCC 絶対 TVT 予測
all_anchor_tvt = []  # ブレンド OOF 評価用: 各行の last_anchor_tvt
all_true_tvt  = []   # ブレンド OOF 評価用: 各行の真の TVT
ncc_stats = []

for i, well_id in enumerate(well_ids_all):
    hw, tw = load_well(well_id, TRAIN_DIR)

    anchor    = hw[hw['TVT_input'].notna()]
    eval_zone = hw[hw['TVT_input'].isna() & hw['TVT'].notna()]

    if len(eval_zone) == 0 or len(anchor) == 0:
        continue

    ncc_tvt, ncc_offset, ncc_score = predict_ncc(hw, tw, anchor, eval_zone)
    ncc_stats.append({'well_id': well_id, 'offset': ncc_offset, 'score': ncc_score})

    feats = build_features(hw, tw, anchor, eval_zone, ncc_offset, ncc_score)
    if feats.empty:
        continue

    last_anchor_tvt = anchor.iloc[-1]['TVT_input']
    drift = eval_zone['TVT'].values - last_anchor_tvt

    all_X.append(feats)
    all_y.append(drift)
    all_groups.extend([well_id] * len(feats))
    all_ncc_tvt.append(ncc_tvt)
    all_anchor_tvt.append(np.full(len(feats), last_anchor_tvt))
    all_true_tvt.append(eval_zone['TVT'].values)

    if (i + 1) % 100 == 0:
        print(f"  Loaded {i+1}/{len(train_files)} wells")

X_train      = pd.concat(all_X, ignore_index=True)
y_train      = np.concatenate(all_y)
groups       = np.array(all_groups)
ncc_tvt_all  = np.concatenate(all_ncc_tvt)
anchor_tvt_all = np.concatenate(all_anchor_tvt)
true_tvt_all = np.concatenate(all_true_tvt)

ncc_df = pd.DataFrame(ncc_stats)
print(f"\nTrain set: {X_train.shape}")
print(f"NCC offset: mean={ncc_df['offset'].mean():.2f}, std={ncc_df['offset'].std():.2f}")
print(f"NCC score : mean={ncc_df['score'].mean():.3f}, min={ncc_df['score'].min():.3f}")

# NCC 単体の OOF 精度
ncc_rmse = np.sqrt(mean_squared_error(true_tvt_all, ncc_tvt_all))
print(f"\nNCC-only RMSE (TVT): {ncc_rmse:.4f}")


# ── XGBoost + GroupKFold(5) ──────────────────────────────────────
print("\nTraining XGBoost with GroupKFold(5)...")

params = {
    'n_estimators'      : 3000,
    'max_depth'         : 6,
    'learning_rate'     : 0.02,
    'subsample'         : 0.8,
    'colsample_bytree'  : 0.8,
    'min_child_weight'  : 10,
    'reg_lambda'        : 1.0,
    'reg_alpha'         : 0.1,
    'tree_method'       : 'hist',
    'random_state'      : 42,
    'n_jobs'            : -1,
}

gkf = GroupKFold(n_splits=5)
oof_drift = np.zeros(len(X_train))
models = []

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
    model = xgb.XGBRegressor(**params, early_stopping_rounds=100, verbosity=0)
    model.fit(
        X_train.iloc[trn_idx], y_train[trn_idx],
        eval_set=[(X_train.iloc[val_idx], y_train[val_idx])],
        verbose=False,
    )
    oof_drift[val_idx] = model.predict(X_train.iloc[val_idx])
    models.append(model)

    fold_rmse = np.sqrt(mean_squared_error(y_train[val_idx], oof_drift[val_idx]))
    print(f"  Fold {fold+1} RMSE (drift): {fold_rmse:.4f}")

# XGB 単体の OOF
xgb_tvt_oof  = anchor_tvt_all + oof_drift
xgb_rmse_tvt = np.sqrt(mean_squared_error(true_tvt_all, xgb_tvt_oof))
print(f"\nXGB-only  OOF RMSE (TVT): {xgb_rmse_tvt:.4f}")

# ブレンド OOF（各 alpha で評価）
print("\nBlend OOF RMSE (TVT) by alpha:")
best_alpha, best_rmse = 0.0, float('inf')
for alpha in np.arange(0.0, 1.05, 0.1):
    blended = alpha * ncc_tvt_all + (1 - alpha) * xgb_tvt_oof
    rmse    = np.sqrt(mean_squared_error(true_tvt_all, blended))
    marker  = " ← best" if rmse < best_rmse else ""
    print(f"  alpha={alpha:.1f}: RMSE={rmse:.4f}{marker}")
    if rmse < best_rmse:
        best_rmse  = rmse
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha:.1f}  (RMSE: {best_rmse:.4f})")
NCC_ALPHA = best_alpha   # 自動的に最良 alpha を採用


# ── テストデータへの推論 ─────────────────────────────────────────
print(f"\nGenerating test predictions (NCC_ALPHA={NCC_ALPHA:.1f})...")
sub = pd.read_csv(SAMPLE_SUB)
test_well_ids = sorted(sub['id'].str.rsplit('_', n=1).str[0].unique())
print(f"Test wells: {test_well_ids}")

all_preds = {}

for well_id in test_well_ids:
    hw, tw = load_well(well_id, TEST_DIR)

    anchor    = hw[hw['TVT_input'].notna()]
    eval_zone = hw[hw['TVT_input'].isna()]

    if len(eval_zone) == 0 or len(anchor) == 0:
        last_tvt = anchor['TVT_input'].iloc[-1] if len(anchor) > 0 else 0.0
        for idx in eval_zone.index:
            all_preds[f'{well_id}_{idx}'] = last_tvt
        continue

    # Pipeline A: NCC
    ncc_tvt, ncc_offset, ncc_score = predict_ncc(hw, tw, anchor, eval_zone)

    # Pipeline B: XGBoost
    last_anchor_tvt = anchor.iloc[-1]['TVT_input']
    feats           = build_features(hw, tw, anchor, eval_zone, ncc_offset, ncc_score)
    drift_preds     = np.mean([m.predict(feats) for m in models], axis=0)
    xgb_tvt         = last_anchor_tvt + drift_preds
    if len(xgb_tvt) >= 17:
        xgb_tvt = savgol_filter(xgb_tvt, window_length=17, polyorder=3)

    # ブレンド
    tvt_preds = NCC_ALPHA * ncc_tvt + (1 - NCC_ALPHA) * xgb_tvt

    for idx, tvt in zip(eval_zone.index, tvt_preds):
        all_preds[f'{well_id}_{idx}'] = tvt

    print(f"  {well_id}: ncc_offset={ncc_offset:.1f}ft, ncc_score={ncc_score:.3f}")

sub['tvt'] = sub['id'].map(all_preds)
print(f"\nNull count: {sub['tvt'].isnull().sum()}")
print(sub.describe())

sub.to_csv('submission.csv', index=False)
print('\nSaved: submission.csv')

# 特徴量重要度 top 10
feat_imp = pd.Series(
    models[0].feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(feat_imp.head(10))
