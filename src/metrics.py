"""
metrics.py — measurement harness for the CDTS pipeline.

Everything downstream (calibration, equity, drift, coupling) imports from here so a
single, tested definition of each metric is reused. Design rules baked in:

  * ECE defaults to EQUAL-MASS bins. Equal-width bins are biased (Roelofs et al. 2022),
    so equal-width is available but never the default.
  * Three ECE estimators are provided to show conclusions are not a binning artefact:
      - ece            : standard L1 ECE (equal-mass)
      - ece_debiased   : finite-sample debiased RMS calibration error (Kumar et al. 2019)
      - ece_sweep      : monotonic-sweep bin selection (Roelofs et al. 2022), L1
  * Every scalar metric is meant to be wrapped in bootstrap_ci(...). No point estimate
    ships without a CI in this project.
  * AUC deltas use DeLong (delong_roc_test) rather than naive bootstrap when a paired,
    analytic test is wanted.

All functions operate on p = predicted P(fake) in [0, 1] and y in {0, 1}.
No dependency beyond numpy / scipy / scikit-learn.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _check(p, y):
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if p.shape != y.shape:
        raise ValueError(f"p and y shape mismatch: {p.shape} vs {y.shape}")
    if p.size == 0:
        raise ValueError("empty input")
    if np.any((p < -_EPS) | (p > 1 + _EPS)):
        raise ValueError("p must lie in [0, 1] (pass probabilities, not logits)")
    return np.clip(p, 0.0, 1.0), y


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _bin_indices(p, n_bins, scheme="equal_mass"):
    """Return a list of index arrays, one per bin."""
    n = p.size
    if scheme == "equal_mass":
        order = np.argsort(p, kind="mergesort")
        return [b for b in np.array_split(order, n_bins) if b.size > 0]
    elif scheme == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        # right-closed bins; last bin includes 1.0
        idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
        return [np.where(idx == b)[0] for b in range(n_bins) if np.any(idx == b)]
    else:
        raise ValueError("scheme must be 'equal_mass' or 'equal_width'")


# --------------------------------------------------------------------------- #
# calibration error estimators
# --------------------------------------------------------------------------- #
def ece(p, y, n_bins=15, scheme="equal_mass"):
    """Standard L1 Expected Calibration Error. Equal-mass by default."""
    p, y = _check(p, y)
    n = p.size
    total = 0.0
    for b in _bin_indices(p, n_bins, scheme):
        conf = p[b].mean()
        acc = y[b].mean()
        total += (b.size / n) * abs(acc - conf)
    return float(total)


def mce(p, y, n_bins=15, scheme="equal_mass"):
    """Maximum Calibration Error: worst per-bin gap."""
    p, y = _check(p, y)
    gaps = [abs(y[b].mean() - p[b].mean()) for b in _bin_indices(p, n_bins, scheme)]
    return float(max(gaps)) if gaps else 0.0


def ece_debiased(p, y, n_bins=15, scheme="equal_mass"):
    """
    Finite-sample debiased RMS calibration error (Kumar et al. 2019).

    The plug-in squared gap per bin is biased upward by acc(1-acc)/(n_b-1); subtract it,
    floor at 0, and return the square root so it reads on the same scale as ECE. On small
    calibration sets this is materially below the naive plug-in, which is the point.
    """
    p, y = _check(p, y)
    n = p.size
    sq = 0.0
    for b in _bin_indices(p, n_bins, scheme):
        nb = b.size
        conf = p[b].mean()
        acc = y[b].mean()
        var = acc * (1 - acc) / max(nb - 1, 1)
        sq += (nb / n) * ((acc - conf) ** 2 - var)
    return float(np.sqrt(max(sq, 0.0)))


def ece_sweep(p, y, scheme="equal_mass", max_bins=50):
    """
    Monotonic-sweep ECE (Roelofs et al. 2022): pick the largest bin count whose per-bin
    accuracies stay monotonically non-decreasing in confidence, then report L1 ECE there.
    Removes the arbitrary fixed-n_bins choice.
    """
    p, y = _check(p, y)
    best_b = 2
    for b in range(2, max_bins + 1):
        bins = _bin_indices(p, b, scheme)
        if len(bins) < b:  # degenerate (ties/empties) — stop growing
            break
        accs = np.array([y[ix].mean() for ix in bins])
        if np.all(np.diff(accs) >= -_EPS):
            best_b = b
        else:
            break
    return ece(p, y, n_bins=best_b, scheme=scheme)


def reliability_curve(p, y, n_bins=15, scheme="equal_mass"):
    """Per-bin (mean_confidence, empirical_accuracy, count) for plotting in notebooks."""
    p, y = _check(p, y)
    conf, acc, cnt = [], [], []
    for b in _bin_indices(p, n_bins, scheme):
        conf.append(p[b].mean()); acc.append(y[b].mean()); cnt.append(b.size)
    return np.array(conf), np.array(acc), np.array(cnt)


# --------------------------------------------------------------------------- #
# proper scoring rules
# --------------------------------------------------------------------------- #
def brier_score(p, y):
    p, y = _check(p, y)
    return float(np.mean((p - y) ** 2))


def nll(p, y):
    """Negative log-likelihood (log loss), clipped for stability."""
    p, y = _check(p, y)
    pc = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))


def roc_auc(p, y):
    p, y = _check(p, y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


# --------------------------------------------------------------------------- #
# bootstrap CI — the workhorse; nothing ships without one
# --------------------------------------------------------------------------- #
def bootstrap_ci(fn, arrays, B=2000, ci=(2.5, 97.5), seed=42, stratify_by=None):
    """
    Paired nonparametric bootstrap CI for any metric.

    fn          : callable taking the resampled arrays positionally, returning a scalar.
    arrays      : list/tuple of equal-length arrays passed to fn in order (e.g. [p, y]).
    stratify_by : optional label array; if given, resample within each class so rare
                  classes (U2R-style) do not vanish from a resample.

    Returns (point, lo, hi) where point is fn on the full data.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays must share length")
    rng = np.random.default_rng(seed)
    point = float(fn(*arrays))

    if stratify_by is not None:
        stratify_by = np.asarray(stratify_by).ravel()
        groups = [np.where(stratify_by == c)[0] for c in np.unique(stratify_by)]

    stats_ = np.empty(B, dtype=float)
    valid = 0
    for i in range(B):
        if stratify_by is not None:
            idx = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups])
        else:
            idx = rng.integers(0, n, size=n)
        try:
            val = fn(*[a[idx] for a in arrays])
        except Exception:
            val = np.nan
        stats_[i] = val
        valid += np.isfinite(val)
    stats_ = stats_[np.isfinite(stats_)]
    if stats_.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats_, ci)
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- #
# DeLong — analytic AUC variance and paired test (Sun & Xu 2014 fast algorithm)
# --------------------------------------------------------------------------- #
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r, :] = _midrank(pos[r, :])
        ty[r, :] = _midrank(neg[r, :])
        tz[r, :] = _midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def _compute_ground_truth_statistics(y):
    order = (-y).argsort()
    label_1_count = int(y.sum())
    return order, label_1_count


def delong_roc_variance(y, p):
    """Return (auc, variance) via DeLong. y in {0,1}, p = score."""
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    order, m = _compute_ground_truth_statistics(y)
    preds_sorted = p[order][np.newaxis, :]
    aucs, cov = _fast_delong(preds_sorted, m)
    return float(aucs[0]), float(cov[0, 0])


def delong_roc_test(y, p1, p2):
    """
    Paired DeLong test that AUC(p1) != AUC(p2) on the same labels y.
    Returns (auc1, auc2, z, two_sided_p).
    """
    y = np.asarray(y, dtype=float).ravel()
    order, m = _compute_ground_truth_statistics(y)
    preds = np.vstack((p1, p2))[:, order]
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0 if aucs[0] == aucs[1] else np.inf * np.sign(aucs[0] - aucs[1])
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
    pval = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(pval)
