# -*- coding: utf-8 -*-
"""What survives the ECE tie-artifact correction:
1. Estimator cross-validation: exact conditional-mean ECE over unique score values vs tie-safe.
2. Raw (uncalibrated) ECE vs competence per detector.
3. Section 4.7 transferred-calibrator recomputation: frozen in-domain calibrator (isotonic AND
   Platt AND beta variants) applied to all 21 Xception generators, ECE measured tie-safe.
4. DFD whole-dataset oracle points, tie-safe."""
import sys, glob
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import calibration as cal, metrics as met

def ece_tiesafe(p, y, n_bins=15, seed=0):
    p = np.clip(np.asarray(p, float).ravel(), 0, 1); y = np.asarray(y, float).ravel()
    jit = np.random.default_rng(seed).random(p.size) * 1e-12
    order = np.argsort(p + jit, kind='mergesort'); tot = 0.0
    for b in np.array_split(order, n_bins):
        if b.size: tot += b.size * abs(p[b].mean() - y[b].mean())
    return tot / p.size

def ece_exact_conditional(p, y):
    """Exact L1 calibration error over unique predicted values: sum_v (n_v/N)|v - mean(y|p=v)|.
    No binning of distinct values; the reference for tie handling."""
    p = np.clip(np.asarray(p, float).ravel(), 0, 1); y = np.asarray(y, float).ravel()
    df = pd.DataFrame({'p': np.round(p, 10), 'y': y})
    g = df.groupby('p')['y'].agg(['mean', 'size'])
    return float((g['size'] * (g.index.values - g['mean']).abs()).sum() / len(df))

def split_gen(df):
    p = df.prob_fake.values.astype(float); y = df.label.values.astype(int)
    g = df.identity_id.astype(str).values
    ci, ti, _ = cal.leakage_safe_split(y, groups=g, calib_frac=0.5, seed=42)
    return p[ci], y[ci], p[ti], y[ti]

# ---------- 1. estimator cross-validation on three tricky configs ----------
print("### 1. tie-safe vs exact-conditional ECE (must agree; paper estimator shown for contrast)")
for f in ['xceptionFR_df40_StyleGAN2', 'xceptionFS_df40_wav2lip', 'effnetb4_df40_pixart']:
    df = pd.read_parquet(f'reports/scores/{f}.parquet')
    pc, yc, pt, yt = split_gen(df)
    ph, _ = cal.fit_predict("hybrid", pc, yc, pt, switch_threshold_n=1000)
    print(f"{f:32s} paper={met.ece(np.clip(ph,0,1), yt, 15, 'equal_mass'):.4f}"
          f"  tiesafe={ece_tiesafe(ph, yt):.4f}  exact={ece_exact_conditional(ph, yt):.4f}")

# ---------- 2. raw-score ECE vs competence ----------
print("\n### 2. RAW (uncalibrated) ECE ~ AUC per detector, tie-safe estimator")
DET = {'xception': ('xceptionFS', 'labelfree_signals.csv'),
       'effnetb4': ('effnetb4', 'unified_trust_signals_effnet.csv'),
       'clip': ('clip', 'unified_trust_signals_clip.csv')}
raw_rows = []
for det, (key, csv) in DET.items():
    gens = pd.read_csv(f'reports/calibration/{csv}').method.tolist()
    pts = []
    for gen in gens:
        df = pd.read_parquet(f'reports/scores/{key}_df40_{gen}.parquet')
        pc, yc, pt, yt = split_gen(df)
        pts.append((roc_auc_score(yt, pt), ece_tiesafe(pt, yt), gen))
    a = np.array([x[0] for x in pts]); e = np.array([x[1] for x in pts])
    r, pv = pearsonr(a, e)
    print(f"  {det:9s} raw-ECE~AUC r = {r:+.3f} (p={pv:.2e}), raw-ECE range {e.min():.3f}-{e.max():.3f}")
    for auc_, ece_, gen_ in pts:
        raw_rows.append(dict(detector=det, method=gen_, AUC=auc_, ECE_raw_tiesafe=ece_))
pd.DataFrame(raw_rows).to_csv('audits/raw_ece_tiesafe.csv', index=False)

# ---------- 3. transferred calibrator (Section 4.7), Xception ----------
print("\n### 3. TRANSFERRED calibrator on Xception (fit once on in-domain FS pool, frozen)")
INDOM = ['simswap', 'blendface', 'facedancer', 'fsgan', 'faceswap', 'inswap']
gens21 = pd.read_csv('reports/calibration/labelfree_signals.csv').method.tolist()
# pooled in-domain calibration split (calib halves of the in-domain generators)
pcs, ycs = [], []
splits = {}
for gen in gens21:
    df = pd.read_parquet(f'reports/scores/xceptionFS_df40_{gen}.parquet')
    pc, yc, pt, yt = split_gen(df)
    splits[gen] = (pt, yt)
    if gen in INDOM:
        pcs.append(pc); ycs.append(yc)
pc_pool = np.concatenate(pcs); yc_pool = np.concatenate(ycs)
print(f"  in-domain pooled calibration set: n={len(pc_pool)}, prevalence={yc_pool.mean():.3f}")

transf = {'iso': cal.IsotonicCalibration().fit(pc_pool, yc_pool),
          'platt': cal.PlattScaling().fit(pc_pool, yc_pool),
          'beta': cal.BetaCalibration().fit(pc_pool, yc_pool)}
rows = []
for gen in gens21:
    pt, yt = splits[gen]
    row = dict(method=gen, AUC=roc_auc_score(yt, pt), ECE_raw=ece_tiesafe(pt, yt))
    for name, c in transf.items():
        row[f'ECE_transf_{name}'] = ece_tiesafe(c.predict(pt), yt)
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv('audits/transferred_tiesafe.csv', index=False)
print(T.round(3).to_string(index=False))
for name in ['iso', 'platt', 'beta']:
    r, pv = pearsonr(T.AUC, T[f'ECE_transf_{name}'])
    out = T[~T.method.isin(INDOM)]
    r_ood, pv_ood = pearsonr(out.AUC, out[f'ECE_transf_{name}'])
    print(f"  transferred-{name:5s}: ECE~AUC r = {r:+.3f} (p={pv:.1e}) all 21; "
          f"r = {r_ood:+.3f} (p={pv_ood:.1e}) on 15 out-of-domain")
lo = T[T.AUC < 0.60]; hi = T[T.AUC > 0.85]
print(f"  transferred-iso mean ECE: AUC>0.85 {hi.ECE_transf_iso.mean():.3f}  vs AUC<0.60 {lo.ECE_transf_iso.mean():.3f}")

# ---------- 4. DFD whole-dataset points, tie-safe ----------
print("\n### 4. DFD whole-dataset oracle points, tie-safe (vs paper 0.127/0.138/0.062)")
import urllib.request
import os
os.makedirs('audits', exist_ok=True)
for f in ['xceptionFS_DFD', 'effnetb4FS_DFD', 'clipFS_DFD']:
    path = f'reports/scores/{f}.parquet'
    import os
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            f'https://raw.githubusercontent.com/anasbiswas1/deepfake-trust-research/main/reports/scores/{f}.parquet', path)
    df = pd.read_parquet(path)
    idcol = 'identity_id' if 'identity_id' in df.columns else ('actor_id' if 'actor_id' in df.columns else 'video_id')
    p = df.prob_fake.values.astype(float); y = df.label.values.astype(int)
    g = df[idcol].astype(str).values
    ci, ti, _ = cal.leakage_safe_split(y, groups=g, calib_frac=0.5, seed=42)
    ph, _ = cal.fit_predict("hybrid", p[ci], y[ci], p[ti], switch_threshold_n=1000)
    pb = cal.BetaCalibration().fit(p[ci], y[ci]).predict(p[ti])
    print(f"  {f:16s} AUC(eval)={roc_auc_score(y[ti], p[ti]):.3f}"
          f"  hybrid: paper-est={met.ece(np.clip(ph,0,1), y[ti], 15, 'equal_mass'):.3f}"
          f" tiesafe={ece_tiesafe(ph, y[ti]):.3f}  beta tiesafe={ece_tiesafe(pb, y[ti]):.3f}"
          f"  raw tiesafe={ece_tiesafe(p[ti], y[ti]):.3f}")
