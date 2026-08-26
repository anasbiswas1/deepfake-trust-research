"""run_routing_figures.py - reproduce the risk-coverage routing results and figures.

Rebuilds the frame pools from reports/scores/*.parquet, recomputes the
risk-coverage curves for the per-frame confidence and batch-dispersion routing
signals, asserts that every published statistic in
reports/calibration/routing_risk_coverage_v2.csv and
reports/calibration/routing_cross_architecture.csv is reproduced (confidence
exactly; dispersion within the published seed noise, AURC_sd = 0.0001), and
writes figures/submission/Fig12.{png,eps} and Fig13.{png,eps}.

The batch order for the dispersion signal is loaded from
reports/calibration/routing_dispersion_orders.json; see the provenance note
there. Curves are drawn from 1% coverage; within-batch order uses
numpy.random.default_rng(0), matching the committed CSVs (seed 0 shown).
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, "reports", "scores")
CAL = os.path.join(HERE, "reports", "calibration")
OUT = os.path.join(HERE, "figures", "submission")
os.makedirs(OUT, exist_ok=True)
TEAL, AMBER, GRAY = "#0f6f74", "#d97b29", "#666666"
POOLS = {"xception": "xceptionFS_df40_*.parquet",
         "effnetb4": "effnetb4_df40_*.parquet",
         "clip": "clip_df40_*.parquet"}

def load_pool(pattern):
    fs = sorted(glob.glob(os.path.join(SCORES, pattern)))
    pool = pd.concat([pd.read_parquet(f)[["method", "label", "prob_fake"]] for f in fs],
                     ignore_index=True)
    pool["mkey"] = pool["method"].str.replace("_cdf", "", regex=False).str.lower()
    pool["error"] = 1 - ((pool["prob_fake"] > 0.5).astype(int) == pool["label"]).astype(int)
    pool["confidence"] = np.maximum(pool["prob_fake"], 1 - pool["prob_fake"])
    return pool

def curve_confidence(pool):
    err = pool["error"].values[np.argsort(-pool["confidence"].values, kind="stable")]
    cov = np.arange(1, len(err) + 1) / len(err)
    return cov, np.cumsum(err) / np.arange(1, len(err) + 1)

def curve_dispersion(pool, order, seed=0):
    parts = [pool[pool.mkey == m].iloc[np.random.default_rng(seed).permutation((pool.mkey == m).sum())]
             for m in order]
    err = pd.concat(parts, ignore_index=True)["error"].values
    cov = np.arange(1, len(err) + 1) / len(err)
    return cov, np.cumsum(err) / np.arange(1, len(err) + 1)

def stats(cov, ce):
    a = np.trapezoid(ce, cov) if hasattr(np, "trapezoid") else np.trapz(ce, cov)
    e = lambda c: ce[min(np.searchsorted(cov, c), len(ce) - 1)]
    return np.array([a, e(0.4), e(0.7), e(0.8), e(0.9)])

def decimate(cov, ce, cmin=0.01, n=1500):
    m = cov >= cmin
    cov, ce = cov[m], ce[m]
    idx = np.unique(np.linspace(0, len(cov) - 1, n).astype(int))
    return cov[idx], ce[idx]

def draw(ax, curves, legend=False):
    (cc, ce), (dc, de), base = curves
    ax.plot(cc, ce, color=TEAL, lw=1.7, ls="-", marker="o", markevery=150, ms=4,
            label="confidence (per-frame): solid, circles")
    ax.plot(dc, de, color=AMBER, lw=1.7, ls="--", marker="^", markevery=(75, 150), ms=4.5,
            label="batch dispersion (reference-free): dashed, triangles")
    ax.axhline(base, color=GRAY, ls=":", lw=1.4, label="no abstention: dotted")
    ax.set_xlabel("Coverage (fraction answered)", fontsize=10.5)
    ax.grid(alpha=0.25); ax.set_xlim(0, 1.02)
    if legend:
        ax.legend(fontsize=8.4, loc="lower right", framealpha=0.95)

def main():
    orders = json.load(open(os.path.join(CAL, "routing_dispersion_orders.json")))["orders"]
    xarch = pd.read_csv(os.path.join(CAL, "routing_cross_architecture.csv"))
    curves = {}
    for arch, pat in POOLS.items():
        pool = load_pool(pat)
        cc, ce = curve_confidence(pool)
        rc = xarch[(xarch.architecture == arch) & (xarch.routing.str.startswith("confidence"))].iloc[0]
        t = np.array([rc.AURC, rc.err40, rc.err70, rc.err80, rc.err90])
        assert np.abs(stats(cc, ce) - t).max() < 6e-4, (arch, "confidence")
        dc, de = curve_dispersion(pool, orders[arch])
        rd = xarch[(xarch.architecture == arch) & (xarch.routing.str.startswith("batch disp"))].iloc[0]
        t = np.array([rd.AURC, rd.err40, rd.err70, rd.err80, rd.err90])
        dev = np.abs(stats(dc, de) - t).max()
        assert dev < 6e-4, (arch, "dispersion", dev)
        print(f"{arch}: confidence exact; dispersion reproduced (max dev {dev:.4f})")
        curves[arch] = (decimate(cc, ce), decimate(dc, de), pool["error"].mean())
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    draw(ax, curves["xception"], legend=True)
    ax.set_ylabel("Selective error", fontsize=10.5)
    plt.tight_layout()
    for ext in ("png", "eps"):
        plt.savefig(os.path.join(OUT, f"Fig12.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, arch, lab in zip(axes, ["xception", "effnetb4", "clip"], ["(a)", "(b)", "(c)"]):
        draw(ax, curves[arch], legend=(arch == "xception"))
        ax.text(0.02, 0.97, lab, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    axes[0].set_ylabel("Selective error", fontsize=10.5)
    plt.tight_layout()
    for ext in ("png", "eps"):
        plt.savefig(os.path.join(OUT, f"Fig13.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("figures written to figures/submission/")

if __name__ == "__main__":
    main()
