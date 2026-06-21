"""
Phase 2-3: Sliding Window NCC (per-point TVT estimation)
目標スコア: < 10.0 (Phase 2-2: OOF 15.09)

改善点:
- Phase 2-2: アンカー区間で1つのNCC offset → eval zone全体に定数適用
- Phase 2-3: eval zone内の各チャンクでローカルGRウィンドウのNCC → per-point offset推定
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
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'train' in dirs:
            DATA_DIR = root
            break

if DATA_DIR is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(_here, '..', 'data')

TRAIN_DIR  = os.path.join(DATA_DIR, 'train')
TEST_DIR   = os.path.join(DATA_DIR, 'test')
SAMPLE_SUB = os.path.join(DATA_DIR, 'sample_submission.csv')
print(f"DATA_DIR: {DATA_DIR}")

SW_CHUNK_SIZE  = 10    # eval zone の何点ごとに1回 NCC を計算するか
SW_WINDOW_HALF = 40    # GR ウィンドウ幅 [MD ft]
SW_SEARCH_RANGE = 80   # NCC 探索範囲 [ft]
SW_STEP         = 2.0  # NCC 探索ステップ [ft]


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


# ── Well-level NCC（アンカー区間による初期位置推定）───────────────
def ncc_align(anchor, tw, search_range=80, step=0.5):
    """アンカー末尾のGRとtypewell GRを照合し、ベストTVTオフセットを返す。"""
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


# ── Sliding Window NCC（per-point TVT推定）──────────────────────
def compute_sw_ncc(hw, tw, anchor, eval_zone):
    """
    eval zone 内の各チャンクに対してローカル GR ウィンドウで NCC を実行。
    per-point の NCC オフセット・スコア・直接TVT予測を返す。
    """
    tw_gr_interp = interp1d(
        tw['TVT'].values, tw['GR'].values,
        bounds_error=False, fill_value='extrapolate'
    )

    last_anchor_tvt = anchor.iloc[-1]['TVT_input']
    last_anchor_md  = anchor.iloc[-1]['MD']
    tail = anchor.tail(100)
    slope = np.polyfit(tail['MD'], tail['TVT_input'], 1)[0] if len(tail) >= 5 else 0.0
    slope = np.clip(slope, -0.02, 0.02)

    n        = len(eval_zone)
    eval_md  = eval_zone['MD'].values
    hw_md    = hw['MD'].values
    hw_gr    = hw['GR'].values

    baseline_tvt = last_anchor_tvt + slope * (eval_md - last_anchor_md)
    offsets_arr  = np.arange(-SW_SEARCH_RANGE, SW_SEARCH_RANGE + SW_STEP, SW_STEP)

    # チャンク中心インデックス（eval zone 内）
    chunk_centers = list(range(SW_CHUNK_SIZE // 2, n, SW_CHUNK_SIZE))
    if not chunk_centers or chunk_centers[-1] < n - 1:
        chunk_centers.append(n - 1)

    chunk_offsets, chunk_scores = [], []

    for c in chunk_centers:
        center_md = eval_md[c]
        mask = (hw_md >= center_md - SW_WINDOW_HALF) & (hw_md <= center_md + SW_WINDOW_HALF)
        local_md = hw_md[mask]
        local_gr = hw_gr[mask]

        if len(local_gr) < 10:
            chunk_offsets.append(0.0)
            chunk_scores.append(0.0)
            continue

        local_gr = smooth_gr(local_gr) if len(local_gr) >= 11 else local_gr.copy()

        # このウィンドウ内の各点の baseline TVT
        local_base = last_anchor_tvt + slope * (local_md - last_anchor_md)

        # 全オフセットをベクトル化して NCC を一括計算 (n_offsets, window)
        tw_gr_mat = np.stack([tw_gr_interp(local_base + o) for o in offsets_arr])

        std_l = local_gr.std()
        if std_l < 1e-6:
            chunk_offsets.append(0.0)
            chunk_scores.append(0.0)
            continue

        lg_n    = (local_gr - local_gr.mean()) / std_l
        tw_stds = tw_gr_mat.std(axis=1)
        valid   = tw_stds > 1e-6

        scores = np.full(len(offsets_arr), -999.0)
        if valid.any():
            tw_means = tw_gr_mat.mean(axis=1, keepdims=True)
            tw_n = np.where(
                valid[:, None],
                (tw_gr_mat - tw_means) / np.where(tw_stds[:, None] > 1e-6, tw_stds[:, None], 1.0),
                0.0
            )
            raw = (lg_n[None] * tw_n).mean(axis=1)
            scores = np.where(valid & np.isfinite(raw), raw, -999.0)

        best = int(np.argmax(scores))
        chunk_offsets.append(float(offsets_arr[best]))
        chunk_scores.append(float(max(scores[best], -1.0)))

    # チャンク結果を per-point に補間
    if len(chunk_centers) > 1:
        pts = np.arange(n)
        sw_off = interp1d(chunk_centers, chunk_offsets, bounds_error=False,
                          fill_value=(chunk_offsets[0], chunk_offsets[-1]))(pts)
        sw_scr = interp1d(chunk_centers, chunk_scores, bounds_error=False,
                          fill_value=(chunk_scores[0], chunk_scores[-1]))(pts)
    else:
        sw_off = np.full(n, chunk_offsets[0])
        sw_scr = np.full(n, chunk_scores[0])

    # オフセット列をスムージング
    if len(sw_off) >= 11:
        sw_off = savgol_filter(sw_off, window_length=11, polyorder=2)

    # 直接TVT予測 = baseline + per-point offset
    sw_tvt = baseline_tvt + sw_off
    if len(sw_tvt) >= 17:
        sw_tvt = savgol_filter(sw_tvt, window_length=17, polyorder=3)

    return sw_off, sw_scr, sw_tvt


# ── 特徴量構築（SW NCC 特徴量を追加）───────────────────────────
def build_features(hw, tw, anchor, eval_zone,
                   ncc_offset=0.0, ncc_score=0.0,
                   sw_ncc_offsets=None, sw_ncc_scores=None, sw_ncc_tvt=None):
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

    full_gr = pd.Series(pd.concat([anchor['GR'], eval_zone['GR']]).values)
    anc_len = len(anchor)
    row_frac = np.linspace(0, 1, n)

    # SW NCC フォールバック（None の場合）
    if sw_ncc_offsets is None:
        sw_ncc_offsets = np.full(n, ncc_offset)
    if sw_ncc_scores is None:
        sw_ncc_scores = np.full(n, ncc_score)
    if sw_ncc_tvt is None:
        sw_ncc_tvt = baseline_tvt + sw_ncc_offsets

    tw_gr_at_sw = tw_interp(baseline_tvt + sw_ncc_offsets)

    feats = pd.DataFrame({
        # 既存特徴量（Phase 2-2 と同じ）
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
        'ncc_offset'        : ncc_offset,       # well-level NCC (定数)
        'ncc_score'         : ncc_score,
        'X'                 : eval_zone['X'].values,
        'Y'                 : eval_zone['Y'].values,
        # ── SW NCC 新規特徴量 ──
        'sw_ncc_offset'     : sw_ncc_offsets,   # per-point NCC オフセット
        'sw_ncc_score'      : sw_ncc_scores,    # per-point NCC 信頼度
        'sw_ncc_drift'      : sw_ncc_tvt - last_anchor_tvt,  # SW NCC が示す drift
        'gr_diff_tw_sw'     : gr_arr - tw_gr_at_sw,           # GR vs typewell@sw_ncc
        'tw_gr_at_sw'       : tw_gr_at_sw,
    })
    return feats


# ── 学習データ構築 ───────────────────────────────────────────────
print("Loading train data...")
train_files  = sorted(glob.glob(os.path.join(TRAIN_DIR, '*__horizontal_well.csv')))
well_ids_all = [os.path.basename(f).replace('__horizontal_well.csv', '') for f in train_files]

all_X, all_y, all_groups = [], [], []
all_sw_tvt    = []   # SW NCC 直接TVT予測（OOF評価用）
all_anchor_tvt = []
all_true_tvt  = []
sw_stats = []

for i, well_id in enumerate(well_ids_all):
    hw, tw = load_well(well_id, TRAIN_DIR)

    anchor    = hw[hw['TVT_input'].notna()]
    eval_zone = hw[hw['TVT_input'].isna() & hw['TVT'].notna()]

    if len(eval_zone) == 0 or len(anchor) == 0:
        continue

    # Well-level NCC（アンカー区間ベース）
    ncc_offset, ncc_score = ncc_align(anchor, tw)

    # Sliding window NCC（per-point）
    sw_off, sw_scr, sw_tvt = compute_sw_ncc(hw, tw, anchor, eval_zone)

    sw_stats.append({
        'well_id': well_id,
        'ncc_offset': ncc_offset,
        'sw_offset_mean': sw_off.mean(),
        'sw_offset_std': sw_off.std(),
        'sw_score_mean': sw_scr.mean(),
    })

    feats = build_features(hw, tw, anchor, eval_zone, ncc_offset, ncc_score,
                           sw_off, sw_scr, sw_tvt)
    if feats.empty:
        continue

    last_anchor_tvt = anchor.iloc[-1]['TVT_input']
    drift = eval_zone['TVT'].values - last_anchor_tvt

    all_X.append(feats)
    all_y.append(drift)
    all_groups.extend([well_id] * len(feats))
    all_sw_tvt.append(sw_tvt)
    all_anchor_tvt.append(np.full(len(feats), last_anchor_tvt))
    all_true_tvt.append(eval_zone['TVT'].values)

    if (i + 1) % 100 == 0:
        print(f"  Loaded {i+1}/{len(train_files)} wells")

X_train       = pd.concat(all_X, ignore_index=True)
y_train       = np.concatenate(all_y)
groups        = np.array(all_groups)
sw_tvt_all    = np.concatenate(all_sw_tvt)
anchor_tvt_all = np.concatenate(all_anchor_tvt)
true_tvt_all  = np.concatenate(all_true_tvt)

sw_df = pd.DataFrame(sw_stats)
print(f"\nTrain set: {X_train.shape}")
print(f"Well-level NCC offset: mean={sw_df['ncc_offset'].mean():.2f}, std={sw_df['ncc_offset'].std():.2f}")
print(f"SW NCC offset (mean/well): mean={sw_df['sw_offset_mean'].mean():.2f}, std_within={sw_df['sw_offset_std'].mean():.2f}")
print(f"SW NCC score: mean={sw_df['sw_score_mean'].mean():.3f}")

# SW NCC 直接予測の RMSE
sw_rmse = np.sqrt(mean_squared_error(true_tvt_all, sw_tvt_all))
print(f"\nSW NCC direct RMSE (TVT): {sw_rmse:.4f}")


# ── XGBoost + GroupKFold(5) ──────────────────────────────────────
print("\nTraining XGBoost with GroupKFold(5)...")

params = {
    'n_estimators'     : 3000,
    'max_depth'        : 6,
    'learning_rate'    : 0.02,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
    'min_child_weight' : 10,
    'reg_lambda'       : 1.0,
    'reg_alpha'        : 0.1,
    'tree_method'      : 'hist',
    'random_state'     : 42,
    'n_jobs'           : -1,
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

xgb_tvt_oof  = anchor_tvt_all + oof_drift
xgb_rmse_tvt = np.sqrt(mean_squared_error(true_tvt_all, xgb_tvt_oof))
print(f"\nXGB-only  OOF RMSE (TVT): {xgb_rmse_tvt:.4f}")

# ブレンド OOF（SW NCC 直接予測 vs XGB）
print("\nBlend OOF RMSE (TVT) by alpha (SW_NCC * alpha + XGB * (1-alpha)):")
best_alpha, best_rmse = 0.0, float('inf')
for alpha in np.arange(0.0, 1.05, 0.1):
    blended = alpha * sw_tvt_all + (1 - alpha) * xgb_tvt_oof
    rmse    = np.sqrt(mean_squared_error(true_tvt_all, blended))
    marker  = " ← best" if rmse < best_rmse else ""
    print(f"  alpha={alpha:.1f}: RMSE={rmse:.4f}{marker}")
    if rmse < best_rmse:
        best_rmse  = rmse
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha:.1f}  (RMSE: {best_rmse:.4f})")
SW_ALPHA = best_alpha


# ── テストデータへの推論 ─────────────────────────────────────────
print(f"\nGenerating test predictions (SW_ALPHA={SW_ALPHA:.1f})...")
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

    ncc_offset, ncc_score = ncc_align(anchor, tw)
    sw_off, sw_scr, sw_tvt = compute_sw_ncc(hw, tw, anchor, eval_zone)

    last_anchor_tvt = anchor.iloc[-1]['TVT_input']
    feats = build_features(hw, tw, anchor, eval_zone, ncc_offset, ncc_score,
                           sw_off, sw_scr, sw_tvt)
    drift_preds = np.mean([m.predict(feats) for m in models], axis=0)
    xgb_tvt = last_anchor_tvt + drift_preds
    if len(xgb_tvt) >= 17:
        xgb_tvt = savgol_filter(xgb_tvt, window_length=17, polyorder=3)

    tvt_preds = SW_ALPHA * sw_tvt + (1 - SW_ALPHA) * xgb_tvt

    for idx, tvt in zip(eval_zone.index, tvt_preds):
        all_preds[f'{well_id}_{idx}'] = tvt

    print(f"  {well_id}: ncc_offset={ncc_offset:.1f}ft, sw_offset_mean={sw_off.mean():.1f}ft, sw_score={sw_scr.mean():.3f}")

sub['tvt'] = sub['id'].map(all_preds)
print(f"\nNull count: {sub['tvt'].isnull().sum()}")
print(sub.describe())

sub.to_csv('submission.csv', index=False)
print('\nSaved: submission.csv')

# 特徴量重要度 top 15
feat_imp = pd.Series(
    models[0].feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)
print("\nTop 15 feature importances:")
print(feat_imp.head(15))
