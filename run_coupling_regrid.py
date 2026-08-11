# -*- coding: utf-8 -*-
"""Coupling recomputation under a calibrator x estimator grid.
Calibrators: hybrid (paper), Platt (unconstrained), beta, orientation-selected isotonic.
Estimators: paper equal-mass ECE (stable-sort ties) vs tie-safe equal-mass ECE (random tie-break, seed 0).
Detectors: Xception 21-gen timeline, EfficientNet 20-gen, CLIP 20-gen.
Also: pooled-32-style recomputation for the FR-mismatch and FF++ configurations, coupling
re-estimates excluding AUC<0.5, using AUC*, and restricted to the 0.6-0.85 band."""
import sys, glob
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import calibration as cal, metrics as met
import os
os.makedirs('audits', exist_ok=True)

RNG = np.random.default_rng(0)

def ece_tiesafe(p, y, n_bins=15, seed=0):
    p = np.clip(np.asarray(p, float).ravel(), 0, 1)
    y = np.asarray(y, float).ravel()
    rng = np.random.default_rng(seed)
    jitter = rng.random(p.size) * 1e-12
    order = np.argsort(p + jitter, kind='mergesort')
    tot = 0.0
    for b in np.array_split(order, n_bins):
        if b.size == 0: continue
        tot += b.size * abs(p[b].mean() - y[b].mean())
    return tot / p.size

def per_generator(df):
    p = df.prob_fake.values.astype(float)
    y = df.label.values.astype(int)
    g = df.identity_id.astype(str).values
    ci, ti, _ = cal.leakage_safe_split(y, groups=g, calib_frac=0.5, seed=42)
    pc, yc, pt, yt = p[ci], y[ci], p[ti], y[ti]
    auc = roc_auc_score(yt, pt)

    preds = {}
    preds['hybrid'], _ = cal.fit_predict("hybrid", pc, yc, pt, switch_threshold_n=1000)
    preds['platt'] = cal.PlattScaling().fit(pc, yc).predict(pt)
    preds['beta'] = cal.BetaCalibration().fit(pc, yc).predict(pt)
    iso_up = cal.IsotonicCalibration().fit(pc, yc)
    iso_fl = cal.IsotonicCalibration().fit(1 - pc, yc)
    e_up = met.ece(np.clip(iso_up.predict(pc), 0, 1), yc, 15, 'equal_mass')
    e_fl = met.ece(np.clip(iso_fl.predict(1 - pc), 0, 1), yc, 15, 'equal_mass')
    preds['iso_oriented'] = iso_fl.predict(1 - pt) if e_fl < e_up else iso_up.predict(pt)

    row = dict(AUC=auc)
    for k, v in preds.items():
        row[f'{k}__paperECE'] = met.ece(np.clip(v, 0, 1), yt, 15, 'equal_mass')
        row[f'{k}__tiesafe'] = ece_tiesafe(v, yt)
    return row

DETECTORS = {
    'xception': ('xceptionFS', pd.read_csv('reports/calibration/labelfree_signals.csv').method.tolist()),
    'effnetb4': ('effnetb4', pd.read_csv('reports/calibration/unified_trust_signals_effnet.csv').method.tolist()),
    'clip': ('clip', pd.read_csv('reports/calibration/unified_trust_signals_clip.csv').method.tolist()),
}

all_rows = []
for det, (key, gens) in DETECTORS.items():
    for gen in gens:
        df = pd.read_parquet(f'reports/scores/{key}_df40_{gen}.parquet')
        r = per_generator(df)
        r['detector'] = det; r['method'] = gen
        all_rows.append(r)
        print(f"{det:9s} {gen:12s} AUC {r['AUC']:.3f}  hybrid {r['hybrid__paperECE']:.3f}/{r['hybrid__tiesafe']:.3f}"
              f"  platt {r['platt__tiesafe']:.3f}  beta {r['beta__tiesafe']:.3f}  isoOr {r['iso_oriented__tiesafe']:.3f}")

D = pd.DataFrame(all_rows)
D.to_csv('audits/coupling_regrid_pergen.csv', index=False)

print("\n" + "=" * 90)
print("COUPLING r (Pearson AUC ~ ECE) per detector, per calibrator x estimator")
print("=" * 90)
variants = ['hybrid__paperECE', 'hybrid__tiesafe', 'iso_oriented__tiesafe', 'platt__tiesafe', 'beta__tiesafe']
summary = []
for det in DETECTORS:
    d = D[D.detector == det]
    line = {'detector': det, 'n': len(d)}
    for v in variants:
        r, p = pearsonr(d.AUC, d[v])
        line[v] = f"{r:+.3f}"
        line[v + '_p'] = f"{p:.1e}"
    summary.append(line)
    print(det, {v: line[v] for v in variants})

print("\nBand / exclusion / AUC* re-estimates (tie-safe, per calibrator), pooled over the three detectors' per-gen points:")
D['AUC_star'] = np.maximum(D.AUC, 1 - D.AUC)
for v in ['hybrid__tiesafe', 'platt__tiesafe', 'beta__tiesafe']:
    full = pearsonr(D.AUC, D[v])
    excl = D[D.AUC >= 0.5]; r_excl = pearsonr(excl.AUC, excl[v])
    star = pearsonr(D.AUC_star, D[v])
    band = D[(D.AUC >= 0.6) & (D.AUC <= 0.85)]; r_band = pearsonr(band.AUC, band[v])
    print(f"  {v:22s} full r={full[0]:+.3f}  excl<0.5 r={r_excl[0]:+.3f} (n={len(excl)})"
          f"  AUC* r={star[0]:+.3f}  band0.6-0.85 r={r_band[0]:+.3f} (n={len(band)}, p={r_band[1]:.3f})")

pd.DataFrame(summary).to_csv('audits/coupling_regrid_summary.csv', index=False)
print("\nsaved coupling_regrid_pergen.csv + coupling_regrid_summary.csv")
