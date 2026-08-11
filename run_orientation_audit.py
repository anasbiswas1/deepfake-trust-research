# -*- coding: utf-8 -*-
"""Orientation-and-calibrator audit (Appendix C), dual-estimator version.
For each low-competence configuration and two controls: AUC, AUC*, Platt slope, and
evaluation-half ECE under BOTH the legacy stable-sort estimator and the tie-safe estimator.
The constant-prevalence pair replicates the Section 3.5 demonstration per configuration.
Requires the tie-safe scheme; on an unpatched metrics.py the local ece_ts below is used."""

import sys, glob
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import calibration as cal, metrics as met
import os
os.makedirs('audits', exist_ok=True)

def ece_ts(p, y, n_bins=15, seed=0):
    p = np.clip(np.asarray(p, float).ravel(), 0, 1); y = np.asarray(y, float).ravel()
    jit = np.random.default_rng(seed).random(p.size) * 1e-12
    order = np.argsort(p + jit, kind='mergesort'); tot = 0.0
    for b in np.array_split(order, n_bins):
        if b.size: tot += b.size * abs(p[b].mean() - y[b].mean())
    return tot / p.size

def ece_leg(p, y):
    scheme = 'equal_mass_legacy' if 'equal_mass_legacy' in open('src/metrics.py').read() else 'equal_mass'
    return met.ece(np.clip(p, 0, 1), y, 15, scheme)

def audit(name, df):
    p = df.prob_fake.values.astype(float); y = df.label.values.astype(int)
    ci, ti, _ = cal.leakage_safe_split(y, groups=df.identity_id.astype(str).values, calib_frac=0.5, seed=42)
    pc, yc, pt, yt = p[ci], y[ci], p[ti], y[ti]
    auc = roc_auc_score(yt, pt)
    pl = cal.PlattScaling().fit(pc, yc)
    iso_up = cal.IsotonicCalibration().fit(pc, yc)
    iso_fl = cal.IsotonicCalibration().fit(1 - pc, yc)
    orient = 'flipped' if ece_leg(iso_fl.predict(1 - pc), yc) < ece_leg(iso_up.predict(pc), yc) else 'increasing'
    iso_pred = iso_fl.predict(1 - pt) if orient == 'flipped' else iso_up.predict(pt)
    hyb, _ = cal.fit_predict("hybrid", pc, yc, pt, switch_threshold_n=1000)
    bt = cal.BetaCalibration().fit(pc, yc).predict(pt)
    const = np.full_like(pt, yc.mean())
    return dict(config=name, AUC=round(auc, 3), AUC_star=round(max(auc, 1 - auc), 3),
                platt_slope=round(float(pl._lr.coef_[0][0]), 3), iso_orientation=orient,
                hybrid_legacy=round(ece_leg(hyb, yt), 3), hybrid_tiesafe=round(ece_ts(hyb, yt), 3),
                iso_or_tiesafe=round(ece_ts(iso_pred, yt), 3),
                platt_tiesafe=round(ece_ts(pl.predict(pt), yt), 3),
                beta_tiesafe=round(ece_ts(bt, yt), 3),
                const_tiesafe=round(ece_ts(const, yt), 3), const_legacy=round(ece_leg(const, yt), 3),
                prev_calib=round(yc.mean(), 3), prev_eval=round(yt.mean(), 3))

rows = []
for f in sorted(glob.glob('reports/scores/xceptionFR_df40_*.parquet')):
    rows.append(audit('XcpFR-DF40 ' + f.split('df40_')[1].replace('.parquet', ''), pd.read_parquet(f)))
fr = pd.read_parquet('reports/scores/xception_FR_ffpp_test.parquet'); reals = fr[fr.label == 0]
for m in ['faceswap', 'face2face', 'neuraltextures', 'deepfakes']:
    rows.append(audit('XcpFR-FF++ ' + m, pd.concat([reals, fr[(fr.label == 1) & (fr.method == m)]]).reset_index(drop=True)))
for gen in ['StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'pixart', 'sadtalker']:
    rows.append(audit('EffNet-DF40 ' + gen, pd.read_parquet(f'reports/scores/effnetb4_df40_{gen}.parquet')))
rows.append(audit('CTRL XcpFS simswap', pd.read_parquet('reports/scores/xceptionFS_df40_simswap.parquet')))
rows.append(audit('CTRL XcpFS wav2lip', pd.read_parquet('reports/scores/xceptionFS_df40_wav2lip.parquet')))
D = pd.DataFrame(rows)
pd.set_option('display.width', 300)
print(D.to_string(index=False))
D.to_csv('audits/orientation_audit.csv', index=False)
print('saved audits/orientation_audit.csv')
