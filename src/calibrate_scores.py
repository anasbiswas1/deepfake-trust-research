"""
calibrate_scores.py — turn raw detector scores into calibrated CDTS trust scores,
measure calibration quality, and surface the forgery-coupling signal.

Run AFTER inference (operates on the saved score table, no GPU needed).

Produces three results:
  R1. Baseline miscalibration of raw scores (the problem)
  R2. Calibration effect with HybridCalibrator / Option D (the fix)
  R3. Forgery-coupling: calibration fit on the in-domain method, applied to each
      method — does calibration hold as forgery type shifts? (the contribution)

All calibration is LEAKAGE-SAFE: split by identity_id (group-disjoint), so no
subject appears in both the calibration-fit and evaluation sets.
"""
import numpy as np
import pandas as pd
import calibration as cal
import metrics as met


# ----------------------------------------------------------------------------
def _ece_fn(p, y):
    """bootstrap_ci-compatible: takes arrays (p, y), returns scalar ECE."""
    return met.ece(p, y, n_bins=15, scheme="equal_mass")


def _summary(p, y, label, B=2000):
    """ECE + Brier with bootstrap CI on ECE. Point comes from bootstrap_ci itself
    so point and CI are always mutually consistent."""
    b = met.brier_score(p, y)
    pt, lo, hi = met.bootstrap_ci(_ece_fn, (p, y), B=B, stratify_by=y)
    return {"set": label, "n": len(y), "ECE": pt, "ECE_lo": lo, "ECE_hi": hi, "Brier": b}


def run_calibration_experiment(scores_path, calib_frac=0.5, seed=42,
                               in_domain_method="faceswap", B=2000):
    """Full calibration experiment on a saved score table.

    scores_path: parquet with columns prob_fake,label,identity_id,method,split
    in_domain_method: the method the checkpoint was trained on (for the coupling contrast)
    """
    df = pd.read_parquet(scores_path).reset_index(drop=True)
    p_raw = df["prob_fake"].values.astype(np.float64)
    y = df["label"].values.astype(int)
    groups = df["identity_id"].values
    methods = df["method"].values

    # ---- leakage-safe identity split (group-disjoint) ----
    calib_idx, test_idx, info = cal.leakage_safe_split(
        y, groups=groups, calib_frac=calib_frac, seed=seed)
    print("=== leakage-safe split (by identity) ===")
    print(f"  mode={info['mode']}  n_calib={info['n_calib']}  n_test={info['n_test']}")
    print(f"  calib_pos_rate={info['calib_pos_rate']:.3f}  test_pos_rate={info['test_pos_rate']:.3f}")

    # =========================================================================
    # R1 + R2: baseline vs hybrid-calibrated (pooled, all methods)
    # =========================================================================
    p_cal_fit, y_cal_fit = p_raw[calib_idx], y[calib_idx]
    p_test, y_test = p_raw[test_idx], y[test_idx]

    # fit hybrid (Option D) on calib, apply to test
    p_test_cal, calibrator = cal.fit_predict(
        "hybrid", p_cal_fit, y_cal_fit, p_test, switch_threshold_n=1000)

    print("\n=== R1/R2: pooled calibration (FF++ all methods, test split) ===")
    rows = []
    rows.append(_summary(p_test, y_test, "raw (uncalibrated)", B=B))
    rows.append(_summary(p_test_cal, y_test, "hybrid-calibrated", B=B))
    res_pooled = pd.DataFrame(rows)
    print(res_pooled.to_string(index=False))
    ece_drop = (res_pooled.iloc[0]["ECE"] - res_pooled.iloc[1]["ECE"])
    print(f"  --> ECE reduced by {ece_drop:.4f} "
          f"({100*ece_drop/max(res_pooled.iloc[0]['ECE'],1e-9):.1f}%)")

    # =========================================================================
    # R3: FORGERY COUPLING — calibration fit on in-domain method, applied per method
    # =========================================================================
    # fit calibration ONLY on in-domain method's calibration identities
    cal_mask = np.zeros(len(df), dtype=bool); cal_mask[calib_idx] = True
    test_mask = np.zeros(len(df), dtype=bool); test_mask[test_idx] = True

    # in-domain calibration-fit pool = (calib split) AND (in_domain method OR real)
    indom_fit = cal_mask & ((methods == in_domain_method) | (y == 0))
    p_fit_id = p_raw[indom_fit]; y_fit_id = y[indom_fit]
    print(f"\n=== R3: forgery coupling (calibration fit on '{in_domain_method}'+real, "
          f"n_fit={indom_fit.sum()}) ===")
    print(f"  applied per-method to the TEST split:")

    # fit once on in-domain
    id_calibrator = cal.get_calibrator("hybrid", switch_threshold_n=1000).fit(p_fit_id, y_fit_id)

    coupling_rows = []
    for m in ["faceswap", "deepfakes", "face2face", "neuraltextures"]:
        # eval = test split, this method's fakes + real
        m_eval = test_mask & ((methods == m) | (y == 0))
        if m_eval.sum() == 0:
            continue
        pm = p_raw[m_eval]; ym = y[m_eval]
        pm_cal = id_calibrator.predict(pm)
        e_raw = met.ece(pm, ym, n_bins=15, scheme="equal_mass")
        e_cal = met.ece(pm_cal, ym, n_bins=15, scheme="equal_mass")
        auc = met.roc_auc(pm, ym) if hasattr(met, "roc_auc") else float("nan")
        tag = "  <- in-domain" if m == in_domain_method else ""
        coupling_rows.append({"method": m, "n": int(m_eval.sum()),
                              "AUC": auc, "ECE_raw": e_raw, "ECE_cal": e_cal})
        print(f"    {m:16s} n={m_eval.sum():5d}  AUC={auc:.3f}  "
              f"ECE_raw={e_raw:.4f}  ECE_cal={e_cal:.4f}{tag}")
    res_coupling = pd.DataFrame(coupling_rows)

    print("\n  INTERPRETATION: if ECE_cal stays low for the in-domain method but grows")
    print("  for reenactment methods, the calibration does NOT transfer across forgery")
    print("  types -> motivates adaptive / forgery-aware trust scoring (CDTS).")

    return {"split_info": info, "pooled": res_pooled, "coupling": res_coupling,
            "calibrator": calibrator}
