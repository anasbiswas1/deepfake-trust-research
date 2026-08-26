"""run_fig5_fig9.py - regenerate submission Fig5 and Fig9 as vector graphics.

Fig5: the three DFD raw-ECE points (identity-disjoint evaluation halves,
leakage_safe_split seed 42) against the 61-configuration raw coupling with
OLS fit and 95% prediction interval. Fig9: ROC curves for label-free
detection of deployed calibration risk (transferred ECE > 0.1, 14/21
positive). Every plotted quantity is asserted against the manuscript
values before drawing. Outputs figures/submission/Fig{5,9}.{png,eps}.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import calibration as cal
import metrics as met

CAL = os.path.join(HERE, "reports", "calibration")
SCORES = os.path.join(HERE, "reports", "scores")
OUT = os.path.join(HERE, "figures", "submission")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "ps.fonttype": 42})
TEAL, AMBER, INK, GRAY, GRID = "#0f6f74", "#d97b29", "#22262a", "#9a9a9a", "#dddddd"

def fig5():
    o = pd.read_csv(os.path.join(CAL, "oracle_corrected_pergen.csv"))
    x, y = o["AUC"].values, o["ECE_raw"].values
    r = np.corrcoef(x, y)[0, 1]
    assert abs(r - (-0.962)) < 0.005, r
    n = len(x)
    b, a = np.polyfit(x, y, 1)
    s2 = np.sum((y - (a + b * x)) ** 2) / (n - 2)
    sx = np.sum((x - x.mean()) ** 2)
    xx = np.linspace(0.40, 1.0, 200)
    se = np.sqrt(s2 * (1 + 1 / n + (xx - x.mean()) ** 2 / sx))
    tc = st.t.ppf(0.975, n - 2)
    targets = {"xceptionFS_DFD.parquet": 0.285,
               "effnetb4FS_DFD.parquet": 0.265,
               "clipFS_DFD.parquet": 0.219}
    pts = []
    style = [("*", TEAL, "Xception", (9, 9), "left"),
             ("s", "#2f7d3f", "EfficientNet-B4", (-9, -15), "right"),
             ("^", "#b3382c", "CLIP-ViT", (9, 9), "left")]
    for (f, tgt), (mk, col, lab, off, ha) in zip(targets.items(), style):
        d = pd.read_parquet(os.path.join(SCORES, f))
        yy = d["label"].values.astype(int)
        p = d["prob_fake"].values.astype(np.float64)
        ci, ti, _ = cal.leakage_safe_split(yy, groups=d["identity_id"].values,
                                           calib_frac=0.5, seed=42)
        auc = roc_auc_score(yy[ti], p[ti])
        ece = met.ece(p[ti], yy[ti], n_bins=15, scheme="equal_mass")
        assert abs(ece - tgt) < 0.0015, (f, ece, tgt)
        pts.append((auc, ece, mk, col, lab, off, ha))
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.fill_between(xx, a + b * xx - tc * se, a + b * xx + tc * se, color="#e8e8e8",
                    lw=0, label="95% prediction interval (61 configs)", zorder=1)
    ax.scatter(x, y, s=16, color="#a8a8a8", edgecolors="none",
               label="61 suite configurations (raw ECE)", zorder=2)
    ax.plot(xx, a + b * xx, ls="--", color=INK, lw=1.5,
            label=f"OLS: raw ECE vs AUC (r = {r:.2f})", zorder=3)
    for auc, ece, mk, col, lab, off, ha in pts:
        ax.scatter([auc], [ece], marker=mk, s=210 if mk == "*" else 115, color=col,
                   edgecolors="black", lw=0.9, zorder=5,
                   label=f"DFD: {lab} (AUC {auc:.2f})")
        ax.annotate(lab, (auc, ece), textcoords="offset points", xytext=off,
                    fontsize=8, color=INK, ha=ha)
    ax.text(0.415, -0.05, "Two DFD points inside the interval;\nEfficientNet below it "
            "(better calibrated\nthan predicted). None exceeds predicted risk.",
            fontsize=7.8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fbf4dd", ec="#c9b972", lw=0.8))
    ax.set_xlabel("Detection competence (AUC)", fontsize=11)
    ax.set_ylabel("Raw ECE (tie-safe)", fontsize=11)
    ax.grid(color=GRID)
    ax.legend(fontsize=7.6, loc="upper right", framealpha=1)
    ax.set_xlim(0.39, 1.01)
    plt.tight_layout()
    for ext in ("png", "eps"):
        plt.savefig(os.path.join(OUT, f"Fig5.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig5 reproduced and written")

def fig9():
    t = pd.read_csv(os.path.join(CAL, "transferred_calibrator_tiesafe.csv"))
    s = pd.read_csv(os.path.join(CAL, "labelfree_signals.csv"))
    m = t.merge(s[["method", "entropy", "ks_vs_ref"]], on="method")
    yv = (m["ECE_transf_iso"] > 0.1).astype(int).values
    assert yv.sum() == 14 and len(yv) == 21, (yv.sum(), len(yv))
    ae = roc_auc_score(yv, m["entropy"])
    ak = roc_auc_score(yv, -m["ks_vs_ref"])
    assert abs(ae - 0.990) < 0.002 and abs(ak - 0.939) < 0.002, (ae, ak)
    fe, te, _ = roc_curve(yv, m["entropy"].values)
    fk, tk, _ = roc_curve(yv, -m["ks_vs_ref"].values)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot(fe, te, color=TEAL, lw=1.9, ls="-", marker="o", ms=4.5,
            label=f"predictive entropy \u2014 portable (ROC-AUC = {ae:.2f}): solid, circles")
    ax.plot(fk, tk, color=AMBER, lw=1.9, ls="--", marker="s", ms=4,
            label=f"KS divergence \u2014 Xception-only (ROC-AUC = {ak:.2f}): dashed, squares")
    ax.plot([0, 1], [0, 1], ls=":", color=GRAY, lw=1.2)
    ax.set_xlabel("False alarm rate", fontsize=11)
    ax.set_ylabel("Detection rate", fontsize=11)
    ax.grid(color=GRID)
    ax.legend(fontsize=7.8, loc="lower right", framealpha=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    for ext in ("png", "eps"):
        plt.savefig(os.path.join(OUT, f"Fig9.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig9 reproduced and written")

if __name__ == "__main__":
    fig5()
    fig9()
