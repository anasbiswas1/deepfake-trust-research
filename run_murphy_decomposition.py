# -*- coding: utf-8 -*-
"""Murphy reliability-resolution decomposition of the target-fitted score's Brier score
(Section 4.2 sharpness measurement). Tie-safe equal-mass bins, hybrid target-fitted
protocol, all 61 suite configurations. Reproduces murphy_decomposition.csv."""
import sys, glob
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import calibration as cal
import os
os.makedirs('audits', exist_ok=True)

def murphy(p, y, n_bins=15, seed=0):
    p = np.clip(np.asarray(p, float).ravel(), 0, 1); y = np.asarray(y, float).ravel()
    jit = np.random.default_rng(seed).random(p.size) * 1e-12
    order = np.argsort(p + jit, kind='mergesort')
    ybar = y.mean(); rel = 0.0; res = 0.0
    for b in np.array_split(order, n_bins):
        if b.size:
            rel += b.size * (p[b].mean() - y[b].mean()) ** 2
            res += b.size * (y[b].mean() - ybar) ** 2
    return rel / p.size, res / p.size, ybar * (1 - ybar)

rows = []
for det, pat in [('xception', 'xceptionFS_df40_*.parquet'),
                 ('effnetb4', 'effnetb4_df40_*.parquet'),
                 ('clip', 'clip_df40_*.parquet')]:
    for f in sorted(glob.glob('reports/scores/' + pat)):
        df = pd.read_parquet(f)
        p = df.prob_fake.values.astype(float); y = df.label.values.astype(int)
        ci, ti, _ = cal.leakage_safe_split(y, groups=df.identity_id.astype(str).values, calib_frac=0.5, seed=42)
        calp, _ = cal.fit_predict("hybrid", p[ci], y[ci], p[ti], switch_threshold_n=1000)
        rel, res, unc = murphy(calp, y[ti])
        rows.append(dict(detector=det, method=f.split('df40_')[1][:-8],
                         AUC=roc_auc_score(y[ti], p[ti]), rel=rel, res=res, unc=unc))
D = pd.DataFrame(rows)
for det, g in D.groupby('detector'):
    print(det, 'r(AUC,res)=%+.3f' % pearsonr(g.AUC, g.res)[0], 'r(AUC,rel)=%+.3f' % pearsonr(g.AUC, g.rel)[0])
print('pooled r(AUC,res)=%+.3f' % pearsonr(D.AUC, D.res)[0])
D.to_csv('audits/murphy_decomposition.csv', index=False)
print('saved audits/murphy_decomposition.csv')
