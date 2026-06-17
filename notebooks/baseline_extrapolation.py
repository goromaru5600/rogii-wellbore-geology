import os
import glob
import numpy as np
import pandas as pd

# データパスを動的に解決
DATA_DIR = '/kaggle/input/competitions/rogii-wellbore-geology-prediction'
if not os.path.exists(DATA_DIR):
    DATA_DIR = '/kaggle/input/rogii-wellbore-geology-prediction'

TEST_DIR   = os.path.join(DATA_DIR, 'test')
SAMPLE_SUB = os.path.join(DATA_DIR, 'sample_submission.csv')

print(f"DATA_DIR: {DATA_DIR}")
print(f"Test files: {sorted(os.listdir(TEST_DIR))}")

# sample_submission から well_id を動的に取得
sub = pd.read_csv(SAMPLE_SUB)
well_ids = sorted(sub['id'].str.rsplit('_', n=1).str[0].unique())
print(f"Well IDs in submission: {well_ids}")

# 各 well について最後の既知 TVT_input から定数外挿
predictions = {}
for well_id in well_ids:
    fpath = os.path.join(TEST_DIR, f'{well_id}__horizontal_well.csv')
    df = pd.read_csv(fpath)

    valid = df[df['TVT_input'].notna()]
    if len(valid) == 0:
        # TVT_input が全 NaN の場合 → typewell の TVT 平均を使用
        tw_path = os.path.join(TEST_DIR, f'{well_id}__typewell.csv')
        tw = pd.read_csv(tw_path)
        fallback_tvt = tw['TVT'].mean()
        print(f"{well_id}: no TVT_input found, using typewell mean={fallback_tvt:.2f}")
        for idx in df.index:
            predictions[f'{well_id}_{idx}'] = fallback_tvt
    else:
        last_tvt = valid.iloc[-1]['TVT_input']
        last_md  = valid.iloc[-1]['MD']

        # 直近50行の傾き（MD に対する TVT の変化率）
        recent = valid.tail(50)
        if len(recent) >= 2:
            slope = np.polyfit(recent['MD'], recent['TVT_input'], 1)[0]
            # 水平区間では傾きは非常に小さいはずなので 0.01 ft/ft でクリップ
            slope = np.clip(slope, -0.01, 0.01)
        else:
            slope = 0.0

        pred_rows = df[df['TVT_input'].isna()]
        for idx, row in pred_rows.iterrows():
            tvt_pred = last_tvt + slope * (row['MD'] - last_md)
            predictions[f'{well_id}_{idx}'] = tvt_pred

        print(f"{well_id}: last_tvt={last_tvt:.2f}, slope={slope:.4f}, pred_rows={len(pred_rows)}")

sub['tvt'] = sub['id'].map(predictions)
null_count = sub['tvt'].isnull().sum()
print(f"\nNull count: {null_count}")
print(f"Shape: {sub.shape}")
print(sub.head())

sub.to_csv('submission.csv', index=False)
print('\nSaved: submission.csv')
