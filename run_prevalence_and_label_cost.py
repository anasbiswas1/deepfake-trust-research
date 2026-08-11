# -*- coding: utf-8 -*-
"""Two controls for Section 4 (Sections 4.1/4.2/4.7).
1) Global-bias and prevalence control: per configuration, the bias component
   |mean score - prevalence|, ECE after a shift-only (intercept-only) correction
   fitted on the calibration half, and raw ECE on class-balanced (50/50)
   evaluation subsamples. Reproduces audits/prevalence_control.csv.
2) Label-efficiency of target-fitted calibration: beta-calibration ECE versus
   calibration-set size (stratified subsamples, 25 seeds) on six configurations
   spanning AUC 0.26-0.94. Reproduces audits/label_efficiency.csv."""
import sys, glob, os
sys.path.insert(0, 'src')
os.makedirs('audits', exist_ok=True)
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from scipy.optimize import minimize_scalar
import calibration as cal

def ece_ts(p, y, n_bins=15, seed=0):
    p = np.clip(np.asarray(p, float).ravel(), 1e-7, 1 - 1e-7); y = np.asarray(y, float).ravel()
    jit = np.random.default_rng(seed).random(p.size) * 1e-12
    order = np.argsort(p + jit, kind='mergesort'); tot = 0.0
    for b in np.array_split(order, n_bins):
        if b.size: tot += b.size * abs(p[b].mean() - y[b].mean())
    return tot / p.size

def logit(p): return np.log(p / (1 - p))
def sig(z): return 1 / (1 + np.exp(-z))

def halves(df):
    p = df.prob_fake.values.astype(float); y = df.label.values.astype(int)
    ci, ti, _ = cal.leakage_safe_split(y, groups=df.identity_id.astype(str).values, calib_frac=0.5, seed=42)
    return (np.clip(p[ci], 1e-7, 1 - 1e-7), y[ci], np.clip(p[ti], 1e-7, 1 - 1e-7), y[ti])

rows = []
for det, pat in [('xception', 'xceptionFS_df40_*.parquet'),
                 ('effnetb4', 'effnetb4_df40_*.parquet'), ('clip', 'clip_df40_*.parquet')]:
    for f in sorted(glob.glob('reports/scores/' + pat)):
        pc, yc, pt, yt = halves(pd.read_parquet(f))
        auc = roc_auc_score(yt, pt)
        zc = logit(pc)
        nll = lambda b: -(yc * np.log(np.clip(sig(zc + b), 1e-9, 1 - 1e-9))
                          + (1 - yc) * np.log(np.clip(1 - sig(zc + b), 1e-9, 1 - 1e-9))).mean()
        b = minimize_scalar(nll, bounds=(-8, 8), method='bounded').x
        rng = np.random.default_rng(0)
        i1 = np.where(yt == 1)[0]; i0 = np.where(yt == 0)[0]; m = min(len(i0), len(i1))
        idx = np.concatenate([rng.choice(i0, m, replace=False), rng.choice(i1, m, replace=False)])
        rows.append(dict(detector=det, method=f.split('df40_')[1][:-8], AUC=auc,
                         ECE_raw=ece_ts(pt, yt), bias_gap=abs(pt.mean() - yt.mean()),
                         ECE_shift_only=ece_ts(sig(logit(pt) + b), yt),
                         ECE_balanced=ece_ts(pt[idx], yt[idx])))
D = pd.DataFrame(rows)
for det, g in D.groupby('detector'):
    print(det, {c: round(pearsonr(g.AUC, g[c])[0], 3)
                for c in ['ECE_raw', 'bias_gap', 'ECE_shift_only', 'ECE_balanced']})
print('median bias share:', round((D.bias_gap / D.ECE_raw).median(), 2))
D.to_csv('audits/prevalence_control.csv', index=False)

CONFIGS = ['xceptionFS_df40_blendface.parquet', 'xceptionFS_df40_ddim.parquet',
           'xceptionFS_df40_DiT.parquet', 'xceptionFS_df40_pixart.parquet',
           'xceptionFR_df40_StyleGAN2.parquet', 'effnetb4_df40_sadtalker.parquet']
res = []
for f in CONFIGS:
    pc, yc, pt, yt = halves(pd.read_parquet('reports/scores/' + f))
    auc = roc_auc_score(yt, pt)
    for n in [25, 50, 100, 250, 1000]:
        eces = []
        for s in range(25):
            rng = np.random.default_rng(s)
            i1 = np.where(yc == 1)[0]; i0 = np.where(yc == 0)[0]
            k0 = max(3, int(round(n * len(i0) / len(yc)))); k1 = n - k0
            if k1 < 3 or k0 > len(i0) or k1 > len(i1): continue
            idx = np.concatenate([rng.choice(i0, k0, replace=False), rng.choice(i1, k1, replace=False)])
            try:
                eces.append(ece_ts(cal.BetaCalibration().fit(pc[idx], yc[idx]).predict(pt), yt))
            except Exception:
                pass
        if eces:
            res.append(dict(config=f, AUC=round(auc, 2), n=n,
                            ece_med=round(float(np.median(eces)), 3),
                            ece_q90=round(float(np.quantile(eces, 0.9)), 3)))
R = pd.DataFrame(res)
print(R.pivot_table(index='config', columns='n', values='ece_med').to_string())
R.to_csv('audits/label_efficiency.csv', index=False)
print('saved audits/prevalence_control.csv, audits/label_efficiency.csv')
