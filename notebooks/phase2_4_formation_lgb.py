"""
Phase 2-4: Formation Features + LightGBM/CatBoost
目標スコア: LB < 7.4 (silver)

核心改善:
- ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA カラムを特徴量として追加
- b_well = TVT_input + Z - formation → formation-based TVT直接予測（RMSE ~0.007）
- テストウェルはtrain CSVからformation値を参照（test well IDs == train well IDs）
- XGBoost → LightGBM(3seed) + CatBoost → Ridge スタッキング
- OOFモデルをテスト推論に再利用（5 fold平均）
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

# ── データパス ────────────────────────────────────────────────────
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

FORMATIONS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
N_SPLITS = 5


# ── ユーティリティ ────────────────────────────────────────────────
def smooth_gr(gr, window=11):
    if len(gr) < window:
        return gr
    return savgol_filter(gr, window_length=window, polyorder=3)


def load_well(well_id, base_dir, train_dir=None):
    hw = pd.read_csv(os.path.join(base_dir, f'{well_id}__horizontal_well.csv'))
    tw = pd.read_csv(os.path.join(base_dir, f'{well_id}__typewell.csv'))
    hw['GR'] = hw['GR'].interpolate(limit_direction='both')
    hw['GR'] = hw['GR'].fillna(hw['GR'].median())

    # Formation列がなければ訓練CSVから補完（test well IDs == train well IDs のため可能）
    if not any(f in hw.columns for f in FORMATIONS):
        if train_dir is not None:
            train_path = os.path.join(train_dir, f'{well_id}__horizontal_well.csv')
            if os.path.exists(train_path):
                tr = pd.read_csv(train_path, usecols=['MD'] + FORMATIONS)
                hw = hw.merge(tr[['MD'] + FORMATIONS], on='MD', how='left')
        if not any(f in hw.columns for f in FORMATIONS):
            for f in FORMATIONS:
                hw[f] = 0.0

    return hw, tw


# ── Well-level NCC ────────────────────────────────────────────────
def ncc_align(anchor, tw, search_range=80, step=0.5):
    tw_interp = interp1d(tw['TVT'].values, tw['GR'].values,
                         bounds_error=False, fill_value='extrapolate')
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
            scores.append(-999.0); continue
        ag, tg = anchor_gr[valid], tw_gr[valid]
        if ag.std() < 1e-6 or tg.std() < 1e-6:
            scores.append(0.0); continue
        score = np.corrcoef(ag, tg)[0, 1]
        scores.append(float(score) if np.isfinite(score) else -999.0)
    best_idx = int(np.argmax(scores))
    return float(offsets[best_idx]), float(scores[best_idx])


# ── Formation-based TVT 直接予測 ──────────────────────────────────
def formation_tvt(anchor, eval_zone):
    """b_well = TVT + Z - ANCC ≈ 定数 → eval zone の TVT を直接推定"""
    last = anchor.iloc[-1]
    b = float(last['TVT_input']) + float(last['Z']) - float(last['ANCC'])
    return b - eval_zone['Z'].values + eval_zone['ANCC'].values


# ── Sliding Window NCC（Phase 2-3から継承） ───────────────────────
SW_CHUNK_SIZE = 10; SW_WINDOW_HALF = 40; SW_SEARCH_RANGE = 80; SW_STEP = 2.0


def compute_sw_ncc(hw, tw, anchor, eval_zone):
    tw_gi = interp1d(tw['TVT'].values, tw['GR'].values,
                     bounds_error=False, fill_value='extrapolate')
    la_tvt = float(anchor.iloc[-1]['TVT_input'])
    la_md  = float(anchor.iloc[-1]['MD'])
    tail = anchor.tail(100)
    slope = np.clip(np.polyfit(tail['MD'], tail['TVT_input'], 1)[0]
                    if len(tail) >= 5 else 0.0, -0.02, 0.02)

    n = len(eval_zone)
    eval_md = eval_zone['MD'].values
    hw_md   = hw['MD'].values
    hw_gr   = hw['GR'].values
    base    = la_tvt + slope * (eval_md - la_md)
    offs    = np.arange(-SW_SEARCH_RANGE, SW_SEARCH_RANGE + SW_STEP, SW_STEP)

    centers = list(range(SW_CHUNK_SIZE // 2, n, SW_CHUNK_SIZE))
    if not centers or centers[-1] < n - 1:
        centers.append(n - 1)

    c_off, c_scr = [], []
    for c in centers:
        cm = eval_md[c]
        mask = (hw_md >= cm - SW_WINDOW_HALF) & (hw_md <= cm + SW_WINDOW_HALF)
        lmd, lgr = hw_md[mask], hw_gr[mask]
        if len(lgr) < 10:
            c_off.append(0.0); c_scr.append(0.0); continue
        lgr = smooth_gr(lgr) if len(lgr) >= 11 else lgr.copy()
        lb  = la_tvt + slope * (lmd - la_md)
        mat = np.stack([tw_gi(lb + o) for o in offs])
        std_l = lgr.std()
        if std_l < 1e-6:
            c_off.append(0.0); c_scr.append(0.0); continue
        lg_n    = (lgr - lgr.mean()) / std_l
        tw_stds = mat.std(axis=1); valid = tw_stds > 1e-6
        sc = np.full(len(offs), -999.0)
        if valid.any():
            tw_n = np.where(valid[:, None],
                            (mat - mat.mean(1, keepdims=True)) /
                            np.where(tw_stds[:, None] > 1e-6, tw_stds[:, None], 1.0), 0.0)
            raw = (lg_n[None] * tw_n).mean(1)
            sc = np.where(valid & np.isfinite(raw), raw, -999.0)
        best = int(np.argmax(sc))
        c_off.append(float(offs[best])); c_scr.append(float(max(sc[best], -1.0)))

    pts = np.arange(n)
    if len(centers) > 1:
        sw_off = interp1d(centers, c_off, bounds_error=False,
                          fill_value=(c_off[0], c_off[-1]))(pts)
        sw_scr = interp1d(centers, c_scr, bounds_error=False,
                          fill_value=(c_scr[0], c_scr[-1]))(pts)
    else:
        sw_off = np.full(n, c_off[0]); sw_scr = np.full(n, c_scr[0])

    if len(sw_off) >= 11:
        sw_off = savgol_filter(sw_off, window_length=11, polyorder=2)
    sw_tvt = base + sw_off
    if len(sw_tvt) >= 17:
        sw_tvt = savgol_filter(sw_tvt, window_length=17, polyorder=3)
    return sw_off, sw_scr, sw_tvt


# ── 特徴量構築 ────────────────────────────────────────────────────
def build_features(hw, tw, anchor, eval_zone,
                   ncc_offset=0.0, ncc_score=0.0,
                   sw_ncc_offsets=None, sw_ncc_scores=None, sw_ncc_tvt=None,
                   form_tvt_pred=None):
    n = len(eval_zone)
    if n == 0:
        return pd.DataFrame()

    tw_interp = interp1d(tw['TVT'].values, tw['GR'].values,
                         bounds_error=False,
                         fill_value=(tw['GR'].values[0], tw['GR'].values[-1]))

    last   = anchor.iloc[-1]
    la_tvt = float(last['TVT_input'])
    la_md  = float(last['MD'])
    la_z   = float(last['Z'])
    la_ancc  = float(last['ANCC'])
    la_astnu = float(last['ASTNU'])
    la_astnl = float(last['ASTNL'])

    tail = anchor.tail(100)
    slope_tvt = np.polyfit(tail['MD'], tail['TVT_input'], 1)[0] if len(tail) >= 5 else 0.0
    anchor_gr_mean = anchor['GR'].tail(50).mean()
    anchor_gr_std  = anchor['GR'].tail(50).std() + 1e-6

    md_arr = eval_zone['MD'].values
    z_arr  = eval_zone['Z'].values
    gr_arr = eval_zone['GR'].values

    md_from_anchor = md_arr - la_md
    z_from_anchor  = z_arr  - la_z
    slope_clipped  = np.clip(slope_tvt, -0.02, 0.02)
    baseline_tvt   = la_tvt + slope_clipped * md_from_anchor

    tw_gr_at_base    = tw_interp(baseline_tvt)
    tw_gr_at_ncc     = tw_interp(baseline_tvt + ncc_offset)
    tw_gr_at_base_m5 = tw_interp(baseline_tvt - 5)
    tw_gr_at_base_p5 = tw_interp(baseline_tvt + 5)
    tw_gr_at_base_m10 = tw_interp(baseline_tvt - 10)
    tw_gr_at_base_p10 = tw_interp(baseline_tvt + 10)

    full_gr  = pd.Series(pd.concat([anchor['GR'], eval_zone['GR']]).values)
    anc_len  = len(anchor)
    row_frac = np.linspace(0, 1, n)

    if sw_ncc_offsets is None: sw_ncc_offsets = np.full(n, ncc_offset)
    if sw_ncc_scores  is None: sw_ncc_scores  = np.full(n, ncc_score)
    if sw_ncc_tvt     is None: sw_ncc_tvt     = baseline_tvt + sw_ncc_offsets
    if form_tvt_pred  is None: form_tvt_pred   = baseline_tvt

    tw_gr_at_sw = tw_interp(baseline_tvt + sw_ncc_offsets)

    ev_ancc  = eval_zone['ANCC'].values
    ev_astnu = eval_zone['ASTNU'].values
    ev_astnl = eval_zone['ASTNL'].values
    ev_egfdu = eval_zone['EGFDU'].values
    ev_egfdl = eval_zone['EGFDL'].values
    ev_buda  = eval_zone['BUDA'].values

    return pd.DataFrame({
        # 位置
        'md_from_anchor'    : md_from_anchor,
        'z_from_anchor'     : z_from_anchor,
        'row_frac'          : row_frac,
        'Z'                 : z_arr,
        'X'                 : eval_zone['X'].values,
        'Y'                 : eval_zone['Y'].values,
        # GR
        'GR'                : gr_arr,
        'gr_norm'           : (gr_arr - anchor_gr_mean) / anchor_gr_std,
        'gr_rm11'           : full_gr.rolling(11,  min_periods=1, center=True).mean().values[anc_len:],
        'gr_rm51'           : full_gr.rolling(51,  min_periods=1, center=True).mean().values[anc_len:],
        'gr_rm151'          : full_gr.rolling(151, min_periods=1, center=True).mean().values[anc_len:],
        'gr_rs11'           : full_gr.rolling(11,  min_periods=1, center=True).std().fillna(0).values[anc_len:],
        'gr_rs51'           : full_gr.rolling(51,  min_periods=1, center=True).std().fillna(0).values[anc_len:],
        # Typewell GR
        'gr_diff_tw'        : gr_arr - tw_gr_at_base,
        'gr_diff_tw_ncc'    : gr_arr - tw_gr_at_ncc,
        'gr_diff_tw_m5'     : gr_arr - tw_gr_at_base_m5,
        'gr_diff_tw_p5'     : gr_arr - tw_gr_at_base_p5,
        'tw_gr_at_base'     : tw_gr_at_base,
        'tw_gr_at_ncc'      : tw_gr_at_ncc,
        'tw_gr_at_base_m10' : tw_gr_at_base_m10,
        'tw_gr_at_base_p10' : tw_gr_at_base_p10,
        # アンカー情報
        'last_anchor_tvt'   : la_tvt,
        'last_anchor_z'     : la_z,
        'slope_tvt'         : slope_tvt,
        'anchor_gr_mean'    : anchor_gr_mean,
        'anchor_gr_std'     : anchor_gr_std,
        # NCC
        'ncc_offset'        : ncc_offset,
        'ncc_score'         : ncc_score,
        # SW NCC
        'sw_ncc_offset'     : sw_ncc_offsets,
        'sw_ncc_score'      : sw_ncc_scores,
        'sw_ncc_drift'      : sw_ncc_tvt - la_tvt,
        'gr_diff_tw_sw'     : gr_arr - tw_gr_at_sw,
        'tw_gr_at_sw'       : tw_gr_at_sw,
        # ─── Formation 特徴量（核心）───
        'ANCC'              : ev_ancc,
        'ASTNU'             : ev_astnu,
        'ASTNL'             : ev_astnl,
        'EGFDU'             : ev_egfdu,
        'EGFDL'             : ev_egfdl,
        'BUDA'              : ev_buda,
        'z_minus_ancc'      : z_arr - ev_ancc,
        'z_minus_astnu'     : z_arr - ev_astnu,
        'z_minus_astnl'     : z_arr - ev_astnl,
        'z_minus_egfdu'     : z_arr - ev_egfdu,
        'z_minus_egfdl'     : z_arr - ev_egfdl,
        'z_minus_buda'      : z_arr - ev_buda,
        'form_astnu_ancc'   : ev_astnu - ev_ancc,
        'form_astnl_astnu'  : ev_astnl - ev_astnu,
        'form_egfdu_astnl'  : ev_egfdu - ev_astnl,
        'form_egfdl_egfdu'  : ev_egfdl - ev_egfdu,
        'form_buda_egfdl'   : ev_buda  - ev_egfdl,
        'b_well_ancc'       : la_tvt + la_z - la_ancc,
        'b_well_astnu'      : la_tvt + la_z - la_astnu,
        'b_well_astnl'      : la_tvt + la_z - la_astnl,
        # Formation-based TVT（最重要）
        'form_tvt_pred'     : form_tvt_pred,
        'form_tvt_drift'    : form_tvt_pred - la_tvt,
    })


# ── 訓練データ構築 ────────────────────────────────────────────────
print("Loading train data...")
train_files  = sorted(glob.glob(os.path.join(TRAIN_DIR, '*__horizontal_well.csv')))
well_ids_all = [os.path.basename(f).replace('__horizontal_well.csv', '') for f in train_files]

all_X, all_y, all_groups = [], [], []
all_form_tvt   = []
all_anchor_tvt = []
all_true_tvt   = []

for i, well_id in enumerate(well_ids_all):
    hw, tw = load_well(well_id, TRAIN_DIR)
    anchor    = hw[hw['TVT_input'].notna()]
    eval_zone = hw[hw['TVT_input'].isna() & hw['TVT'].notna()]
    if len(eval_zone) == 0 or len(anchor) == 0:
        continue

    ncc_offset, ncc_score = ncc_align(anchor, tw)
    sw_off, sw_scr, sw_tvt = compute_sw_ncc(hw, tw, anchor, eval_zone)
    f_tvt = formation_tvt(anchor, eval_zone)

    feats = build_features(hw, tw, anchor, eval_zone, ncc_offset, ncc_score,
                           sw_off, sw_scr, sw_tvt, f_tvt)
    if feats.empty:
        continue

    la_tvt = float(anchor.iloc[-1]['TVT_input'])
    drift  = eval_zone['TVT'].values - la_tvt

    all_X.append(feats)
    all_y.append(drift)
    all_groups.extend([well_id] * len(feats))
    all_form_tvt.append(f_tvt)
    all_anchor_tvt.append(np.full(len(feats), la_tvt))
    all_true_tvt.append(eval_zone['TVT'].values)

    if (i + 1) % 100 == 0:
        print(f"  Loaded {i+1}/{len(train_files)} wells")

X_train        = pd.concat(all_X, ignore_index=True)
y_train        = np.concatenate(all_y)
groups         = np.array(all_groups)
form_tvt_all   = np.concatenate(all_form_tvt)
anchor_tvt_all = np.concatenate(all_anchor_tvt)
true_tvt_all   = np.concatenate(all_true_tvt)

form_rmse = np.sqrt(mean_squared_error(true_tvt_all, form_tvt_all))
print(f"\nTrain set: {X_train.shape}")
print(f"Formation-based TVT RMSE (train): {form_rmse:.4f}")


# ── LightGBM + CatBoost + GroupKFold(5) ──────────────────────────
print("\nTraining models with GroupKFold(5)...")

LGB_BASE = dict(
    num_leaves=255, min_child_samples=15,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=3.0, reg_alpha=0.05, objective='regression',
    verbose=-1, n_jobs=-1,
)
LGB_CONFIGS = [
    dict(learning_rate=0.025, n_estimators=8000, seed=42),
    dict(learning_rate=0.020, n_estimators=8000, seed=7),
    dict(learning_rate=0.030, n_estimators=8000, seed=123),
]
CB_PARAMS = dict(
    iterations=8000, learning_rate=0.025, depth=7, l2_leaf_reg=2.0,
    min_data_in_leaf=15, border_count=254, loss_function='RMSE',
    random_seed=42, od_type='Iter', od_wait=300, verbose=0,
)

gkf = GroupKFold(n_splits=N_SPLITS)
splits = list(gkf.split(X_train, y_train, groups))

# モデルをfoldごとに保存（テスト推論に再利用）
all_models = {}
all_oof    = {}

# LightGBM (3 configs)
for ci, cfg in enumerate(LGB_CONFIGS):
    n_est = cfg['n_estimators']
    params = dict(LGB_BASE, learning_rate=cfg['learning_rate'], seed=cfg['seed'])
    oof = np.zeros(len(X_train), np.float32)
    fold_models = []

    for fold, (trn_idx, val_idx) in enumerate(splits):
        dtr = lgb.Dataset(X_train.iloc[trn_idx], label=y_train[trn_idx])
        dva = lgb.Dataset(X_train.iloc[val_idx], label=y_train[val_idx], reference=dtr)
        m = lgb.train(params, dtr, valid_sets=[dva], num_boost_round=n_est,
                      callbacks=[lgb.early_stopping(250, verbose=False),
                                 lgb.log_evaluation(1000)])
        oof[val_idx] = m.predict(X_train.iloc[val_idx],
                                  num_iteration=m.best_iteration).astype(np.float32)
        fold_models.append(m)
        fold_rmse = np.sqrt(mean_squared_error(y_train[val_idx], oof[val_idx]))
        print(f"  LGB{ci} Fold {fold+1}: {fold_rmse:.4f}")

    oof_rmse = np.sqrt(mean_squared_error(y_train, oof))
    print(f"  LGB{ci} OOF RMSE: {oof_rmse:.4f}\n")
    key = f'lgb{ci}'
    all_models[key] = fold_models
    all_oof[key] = oof

# CatBoost
oof_cb = np.zeros(len(X_train), np.float32)
cb_models = []
for fold, (trn_idx, val_idx) in enumerate(splits):
    m = CatBoostRegressor(**CB_PARAMS)
    m.fit(Pool(X_train.iloc[trn_idx].values, label=y_train[trn_idx]),
          eval_set=Pool(X_train.iloc[val_idx].values, label=y_train[val_idx]),
          use_best_model=True)
    oof_cb[val_idx] = m.predict(X_train.iloc[val_idx].values).astype(np.float32)
    cb_models.append(m)
    fold_rmse = np.sqrt(mean_squared_error(y_train[val_idx], oof_cb[val_idx]))
    print(f"  CB Fold {fold+1}: {fold_rmse:.4f}")

oof_cb_rmse = np.sqrt(mean_squared_error(y_train, oof_cb))
print(f"  CB OOF RMSE: {oof_cb_rmse:.4f}\n")
all_models['cb'] = cb_models
all_oof['cb'] = oof_cb

# Ridge スタッキング（OOFでweightを決定）
Sx = np.column_stack([all_oof[k] for k in all_oof])
ridge = Ridge(alpha=1., fit_intercept=False, positive=True)
ridge.fit(Sx, y_train)
oof_stk = ridge.predict(Sx)
oof_avg = Sx.mean(axis=1)
rmse_stk = np.sqrt(mean_squared_error(y_train, oof_stk))
rmse_avg = np.sqrt(mean_squared_error(y_train, oof_avg))
wts = ridge.coef_ / max(ridge.coef_.sum(), 1e-9)
print(f"Avg OOF: {rmse_avg:.4f} | Ridge Stack OOF: {rmse_stk:.4f}")
print(f"Weights: {dict(zip(all_oof.keys(), wts.round(4)))}")

final_oof_drift = oof_stk if rmse_stk < rmse_avg else oof_avg
final_oof_tvt   = anchor_tvt_all + final_oof_drift

# Formation TVT vs Model ブレンド評価
print("\nBlend OOF RMSE (form_tvt * alpha + model * (1-alpha)):")
best_alpha, best_blend_rmse = 0.0, float('inf')
for alpha in np.arange(0.0, 1.05, 0.1):
    blended = alpha * form_tvt_all + (1 - alpha) * final_oof_tvt
    rmse = np.sqrt(mean_squared_error(true_tvt_all, blended))
    marker = " <- best" if rmse < best_blend_rmse else ""
    print(f"  alpha={alpha:.1f}: RMSE={rmse:.4f}{marker}")
    if rmse < best_blend_rmse:
        best_blend_rmse = rmse
        best_alpha = alpha

print(f"\nBest blend alpha: {best_alpha:.1f} (RMSE: {best_blend_rmse:.4f})")
BLEND_ALPHA = best_alpha


# ── テスト推論 ────────────────────────────────────────────────────
def predict_test_well(well_id):
    hw, tw = load_well(well_id, TEST_DIR, train_dir=TRAIN_DIR)
    anchor    = hw[hw['TVT_input'].notna()]
    eval_zone = hw[hw['TVT_input'].isna()]
    if len(eval_zone) == 0 or len(anchor) == 0:
        return {}, []

    ncc_offset, ncc_score = ncc_align(anchor, tw)
    sw_off, sw_scr, sw_tvt = compute_sw_ncc(hw, tw, anchor, eval_zone)
    f_tvt = formation_tvt(anchor, eval_zone)

    feats = build_features(hw, tw, anchor, eval_zone, ncc_offset, ncc_score,
                           sw_off, sw_scr, sw_tvt, f_tvt)
    la_tvt = float(anchor.iloc[-1]['TVT_input'])

    # 各モデルの予測 (5 fold 平均)
    model_preds = {}
    for key, fold_models in all_models.items():
        if key.startswith('lgb'):
            preds = np.mean([m.predict(feats, num_iteration=m.best_iteration)
                             for m in fold_models], axis=0)
        else:  # cb
            preds = np.mean([m.predict(feats.values) for m in fold_models], axis=0)
        model_preds[key] = preds.astype(np.float32)

    # Ridge スタッキング
    test_Sx = np.column_stack([model_preds[k] for k in all_oof])
    stk_drift = ridge.predict(test_Sx)
    avg_drift = test_Sx.mean(axis=1)
    final_drift = stk_drift if rmse_stk < rmse_avg else avg_drift

    model_tvt = la_tvt + final_drift

    # Formation TVT とのブレンド
    tvt_final = BLEND_ALPHA * f_tvt + (1 - BLEND_ALPHA) * model_tvt
    if len(tvt_final) >= 17:
        tvt_final = savgol_filter(tvt_final, window_length=17, polyorder=3)

    preds_dict = {}
    for idx, tvt in zip(eval_zone.index, tvt_final):
        preds_dict[f'{well_id}_{idx}'] = tvt

    return preds_dict, f_tvt


print(f"\nGenerating test predictions (BLEND_ALPHA={BLEND_ALPHA:.1f})...")
sub = pd.read_csv(SAMPLE_SUB)
test_well_ids = sorted(sub['id'].str.rsplit('_', n=1).str[0].unique())
print(f"Test wells: {test_well_ids}")

all_preds = {}
for well_id in test_well_ids:
    preds_dict, f_tvt = predict_test_well(well_id)
    all_preds.update(preds_dict)
    print(f"  {well_id}: {len(preds_dict)} rows predicted, "
          f"form_tvt range=[{f_tvt.min():.1f}, {f_tvt.max():.1f}]")

sub['tvt'] = sub['id'].map(all_preds)
print(f"\nNull count: {sub['tvt'].isnull().sum()}")
print(sub.describe())
sub.to_csv('submission.csv', index=False)
print('\nSaved: submission.csv')

# 特徴量重要度（LGB0、gain）
lgb0_imp = pd.Series(
    np.mean([m.feature_importance(importance_type='gain')
             for m in all_models['lgb0']], axis=0),
    index=X_train.columns
).sort_values(ascending=False)
print("\nTop 15 feature importances (LGB0, gain):")
print(lgb0_imp.head(15))
