"""
Phase 2-1: Drift Target + Typewell GR Features + XGBoost
目標スコア: < 15 (現在: 28.341)

主要改善点:
1. ターゲットを drift (TVT - last_anchor_tvt) に変換
2. typewell GR との差分特徴量
3. GroupKFold(5) by well_id で適切な CV
4. Savitzky-Golay post-processing
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
if os.path.exists('/kaggle/input/competitions/rogii-wellbore-geology-prediction'):
    DATA_DIR = '/kaggle/input/competitions/rogii-wellbore-geology-prediction'
else:
    # ローカル: スクリプトの場所に関わらず data/ を探す
    _here = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(_here, '..', 'data')

TRAIN_DIR  = os.path.join(DATA_DIR, 'train')
TEST_DIR   = os.path.join(DATA_DIR, 'test')
SAMPLE_SUB = os.path.join(DATA_DIR, 'sample_submission.csv')

print(f"DATA_DIR: {DATA_DIR}")


# ── ユーティリティ ───────────────────────────────────────────────
def load_well(well_id, base_dir):
    """水平井 + typewell を読み込み、GR を補間して返す。"""
    hw = pd.read_csv(os.path.join(base_dir, f'{well_id}__horizontal_well.csv'))
    tw = pd.read_csv(os.path.join(base_dir, f'{well_id}__typewell.csv'))
    # GR の NaN を線形補間（粒子フィルターなどが GR を必要とするため）
    hw['GR'] = hw['GR'].interpolate(limit_direction='both')
    hw['GR'] = hw['GR'].fillna(tw['GR'].mean())
    return hw, tw


def build_features(hw, tw, anchor, eval_zone):
    """
    eval_zone の各行に対して特徴量を構築する。
    typewell は GR の参照に使用。
    """
    n = len(eval_zone)
    if n == 0:
        return pd.DataFrame()

    # typewell GR 補間関数
    tw_tvt = tw['TVT'].values
    tw_gr  = tw['GR'].values
    tw_interp = interp1d(tw_tvt, tw_gr, bounds_error=False,
                         fill_value=(tw_gr[0], tw_gr[-1]))

    # アンカー区間から統計量を取得
    last_anchor      = anchor.iloc[-1]
    last_anchor_tvt  = last_anchor['TVT_input']
    last_anchor_md   = last_anchor['MD']
    last_anchor_z    = last_anchor['Z']

    # アンカー末尾の傾き (TVT_input vs MD)
    tail = anchor.tail(100)
    if len(tail) >= 5:
        slope_tvt = np.polyfit(tail['MD'], tail['TVT_input'], 1)[0]
    else:
        slope_tvt = 0.0

    # アンカー末尾の GR 統計
    anchor_gr_mean = anchor['GR'].tail(50).mean()
    anchor_gr_std  = anchor['GR'].tail(50).std() + 1e-6

    # eval_zone の MD からアンカーまでの距離
    md_arr  = eval_zone['MD'].values
    z_arr   = eval_zone['Z'].values
    gr_arr  = eval_zone['GR'].values

    md_from_anchor = md_arr - last_anchor_md
    z_from_anchor  = z_arr  - last_anchor_z

    # ベースライン TVT 推定（アンカー末尾からの外挿、傾きは小さめにクリップ）
    slope_clipped  = np.clip(slope_tvt, -0.02, 0.02)
    baseline_tvt   = last_anchor_tvt + slope_clipped * md_from_anchor

    # typewell GR 参照特徴量
    tw_gr_at_base   = tw_interp(baseline_tvt)
    tw_gr_at_base_m5 = tw_interp(baseline_tvt - 5)
    tw_gr_at_base_p5 = tw_interp(baseline_tvt + 5)
    tw_gr_at_base_m10 = tw_interp(baseline_tvt - 10)
    tw_gr_at_base_p10 = tw_interp(baseline_tvt + 10)

    gr_diff_tw      = gr_arr - tw_gr_at_base
    gr_diff_tw_m5   = gr_arr - tw_gr_at_base_m5
    gr_diff_tw_p5   = gr_arr - tw_gr_at_base_p5

    # GR ローリング統計（well 全体で計算してから eval_zone 行を取る）
    full_gr = pd.Series(
        pd.concat([anchor['GR'], eval_zone['GR']]).values
    )
    gr_roll_mean11  = full_gr.rolling(11,  min_periods=1, center=True).mean().values
    gr_roll_mean51  = full_gr.rolling(51,  min_periods=1, center=True).mean().values
    gr_roll_mean151 = full_gr.rolling(151, min_periods=1, center=True).mean().values
    gr_roll_std11   = full_gr.rolling(11,  min_periods=1, center=True).std().fillna(0).values
    gr_roll_std51   = full_gr.rolling(51,  min_periods=1, center=True).std().fillna(0).values

    anc_len = len(anchor)
    gr_rm11  = gr_roll_mean11[anc_len:]
    gr_rm51  = gr_roll_mean51[anc_len:]
    gr_rm151 = gr_roll_mean151[anc_len:]
    gr_rs11  = gr_roll_std11[anc_len:]
    gr_rs51  = gr_roll_std51[anc_len:]

    # eval_zone 内の相対位置
    row_frac = np.linspace(0, 1, n)

    feats = pd.DataFrame({
        # 位置系
        'md_from_anchor'    : md_from_anchor,
        'z_from_anchor'     : z_from_anchor,
        'row_frac'          : row_frac,
        'Z'                 : z_arr,
        # GR 生値
        'GR'                : gr_arr,
        'gr_norm'           : (gr_arr - anchor_gr_mean) / anchor_gr_std,
        # GR ローリング統計
        'gr_rm11'           : gr_rm11,
        'gr_rm51'           : gr_rm51,
        'gr_rm151'          : gr_rm151,
        'gr_rs11'           : gr_rs11,
        'gr_rs51'           : gr_rs51,
        # typewell GR 差分
        'gr_diff_tw'        : gr_diff_tw,
        'gr_diff_tw_m5'     : gr_diff_tw_m5,
        'gr_diff_tw_p5'     : gr_diff_tw_p5,
        'tw_gr_at_base'     : tw_gr_at_base,
        'tw_gr_at_base_m10' : tw_gr_at_base_m10,
        'tw_gr_at_base_p10' : tw_gr_at_base_p10,
        # アンカー統計
        'last_anchor_tvt'   : last_anchor_tvt,
        'last_anchor_z'     : last_anchor_z,
        'slope_tvt'         : slope_tvt,
        'anchor_gr_mean'    : anchor_gr_mean,
        'anchor_gr_std'     : anchor_gr_std,
        # 座標
        'X'                 : eval_zone['X'].values,
        'Y'                 : eval_zone['Y'].values,
    })

    return feats


# ── 学習データ構築 ───────────────────────────────────────────────
print("Loading train data...")
train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, '*__horizontal_well.csv')))
well_ids_all = [f.split('/')[-1].replace('__horizontal_well.csv', '') for f in train_files]

all_X, all_y, all_groups = [], [], []

for i, (f, well_id) in enumerate(zip(train_files, well_ids_all)):
    hw, tw = load_well(well_id, TRAIN_DIR)

    anchor    = hw[hw['TVT_input'].notna()]
    eval_zone = hw[hw['TVT_input'].isna() & hw['TVT'].notna()]

    if len(eval_zone) == 0 or len(anchor) == 0:
        continue

    feats = build_features(hw, tw, anchor, eval_zone)
    if feats.empty:
        continue

    last_anchor_tvt = anchor.iloc[-1]['TVT_input']
    drift = eval_zone['TVT'].values - last_anchor_tvt

    all_X.append(feats)
    all_y.append(drift)
    all_groups.extend([well_id] * len(feats))

    if (i + 1) % 100 == 0:
        print(f"  Loaded {i+1}/{len(train_files)} wells")

X_train = pd.concat(all_X, ignore_index=True)
y_train = np.concatenate(all_y)
groups  = np.array(all_groups)

print(f"Train set: {X_train.shape}, target drift range: [{y_train.min():.2f}, {y_train.max():.2f}]")


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
oof_preds = np.zeros(len(X_train))
models = []

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
    model = xgb.XGBRegressor(**params, early_stopping_rounds=100, verbosity=0)
    model.fit(
        X_train.iloc[trn_idx], y_train[trn_idx],
        eval_set=[(X_train.iloc[val_idx], y_train[val_idx])],
        verbose=False,
    )
    oof_preds[val_idx] = model.predict(X_train.iloc[val_idx])
    models.append(model)

    fold_rmse = np.sqrt(mean_squared_error(y_train[val_idx], oof_preds[val_idx]))
    print(f"  Fold {fold+1} RMSE (drift): {fold_rmse:.4f}")

oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"\nOOF RMSE (drift): {oof_rmse:.4f}")
print(f"OOF MSE  (drift): {oof_rmse**2:.4f}")


# ── テストデータへの推論 ─────────────────────────────────────────
print("\nGenerating test predictions...")
sub = pd.read_csv(SAMPLE_SUB)
test_well_ids = sorted(sub['id'].str.rsplit('_', n=1).str[0].unique())
print(f"Test wells: {test_well_ids}")

all_preds = {}

for well_id in test_well_ids:
    hw, tw = load_well(well_id, TEST_DIR)

    anchor    = hw[hw['TVT_input'].notna()]
    eval_zone = hw[hw['TVT_input'].isna()]

    if len(eval_zone) == 0 or len(anchor) == 0:
        # フォールバック: 最後の TVT_input を定数として使用
        last_tvt = anchor['TVT_input'].iloc[-1] if len(anchor) > 0 else 0.0
        for idx in eval_zone.index:
            all_preds[f'{well_id}_{idx}'] = last_tvt
        continue

    feats = build_features(hw, tw, anchor, eval_zone)
    last_anchor_tvt = anchor.iloc[-1]['TVT_input']

    # 5 fold モデルの平均
    drift_preds = np.mean([m.predict(feats) for m in models], axis=0)
    tvt_preds   = last_anchor_tvt + drift_preds

    # Savitzky-Golay スムージング（窓幅 17、3次多項式）
    if len(tvt_preds) >= 17:
        tvt_preds = savgol_filter(tvt_preds, window_length=17, polyorder=3)

    for idx, tvt in zip(eval_zone.index, tvt_preds):
        all_preds[f'{well_id}_{idx}'] = tvt

    print(f"  {well_id}: {len(eval_zone)} rows, drift pred range [{drift_preds.min():.2f}, {drift_preds.max():.2f}]")

sub['tvt'] = sub['id'].map(all_preds)
null_count = sub['tvt'].isnull().sum()
print(f"\nNull count: {null_count}")
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
