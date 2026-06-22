"""
calibration.py — post-hoc calibrators and the leakage-safe data discipline they depend on.

Two things in here are load-bearing for the whole paper:

1. leakage_safe_split / assert_three_way_disjoint
   The X-IDS NB03 v1 bug was a calibration set that overlapped the evaluation set, which
   silently flatters ECE. For deepfakes the same bug hides one layer deeper: frames from the
   SAME identity/video are correlated, so a split that is row-disjoint can still leak if the
   same person appears in both calibration and test. leakage_safe_split therefore supports
   GROUP-disjoint splitting (pass identity/video ids as `groups`) and HARD-ASSERTS disjointness
   rather than trusting it. assert_three_way_disjoint additionally checks the calibration set
   never touches the backbone's TRAINING ids.

2. HybridCalibrator
   The locked rule from config/experiment.yaml: n_calib < switch_threshold_n -> Platt (sigmoid,
   low variance on tiny sets); otherwise isotonic (flexible, lower bias with data). It records
   which branch fired in `.method_` so the choice is auditable per cell.

All calibrators are implemented manually (sklearn primitives only). CalibratedClassifierCV is
deliberately avoided — it raised a classifier error in EXHEART and was bypassed there too.

Calibrator contract:  fit(p, y) -> self ;  predict(p) -> calibrated P(fake) in [0, 1]
where p = raw model P(fake) in [0, 1], y in {0, 1}. Temperature scaling consumes logit(p)
internally so the external interface stays uniform (always pass probabilities).

Note on the 9-method sweep in experiment.yaml: for a BINARY real/fake score the meaningful
set is {uncalibrated, platt, isotonic, temperature, beta, histogram, bbq}. 'dirichlet' is the
MULTICLASS generalization of beta calibration (Kull et al. 2019) and reduces to beta in the
binary case, so it is only relevant if calibration later moves to per-manipulation multiclass.
'spline' (Gupta et al. 2021) has a separate reference implementation and is not wired here yet.
Both are intentionally omitted rather than approximated incorrectly.
"""

from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from scipy.optimize import minimize_scalar

from metrics import logit, sigmoid, nll, _EPS

DEFAULT_SWITCH_THRESHOLD_N = 1000  # mirrors config/experiment.yaml: calibration.switch_threshold_n


# --------------------------------------------------------------------------- #
# leakage-safe splitting (the NB03 guardrail)
# --------------------------------------------------------------------------- #
def leakage_safe_split(y, groups=None, calib_frac=0.5, seed=42):
    """
    Split a held-out pool into (calib_idx, test_idx).

    groups : optional identity/video id per row. When supplied the split is GROUP-disjoint
             (no identity appears in both halves); label balance across groups is then
             approximate and reported via the returned dict. When None, the split is exactly
             label-stratified.

    Returns (calib_idx, test_idx, info). Raises AssertionError if disjointness is violated.
    """
    y = np.asarray(y).ravel()
    n = y.size
    idx_all = np.arange(n)

    if groups is None:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=calib_frac, random_state=seed)
        calib_idx, test_idx = next(splitter.split(idx_all, y))
        mode = "stratified_row"
    else:
        groups = np.asarray(groups).ravel()
        if groups.size != n:
            raise ValueError("groups must align with y")
        splitter = GroupShuffleSplit(n_splits=1, train_size=calib_frac, random_state=seed)
        calib_idx, test_idx = next(splitter.split(idx_all, y, groups))
        mode = "group_disjoint"

    calib_idx = np.sort(calib_idx)
    test_idx = np.sort(test_idx)

    # HARD assertions — the whole reason this function exists.
    assert len(np.intersect1d(calib_idx, test_idx)) == 0, "row overlap between calib and test"
    if groups is not None:
        gc, gt = set(groups[calib_idx]), set(groups[test_idx])
        assert gc.isdisjoint(gt), "identity/group leaked across calib and test"

    info = {
        "mode": mode,
        "n_calib": int(calib_idx.size),
        "n_test": int(test_idx.size),
        "calib_pos_rate": float(np.mean(y[calib_idx])) if calib_idx.size else float("nan"),
        "test_pos_rate": float(np.mean(y[test_idx])) if test_idx.size else float("nan"),
    }
    return calib_idx, test_idx, info


def assert_three_way_disjoint(train_ids, calib_ids, test_ids):
    """
    Verify the backbone TRAINING ids never appear in the calibration or test ids, and that
    calib and test are disjoint. Pass identity/video ids (not row indices). Call this in NB03
    before fitting any calibrator. Raises AssertionError on any leak.
    """
    tr, ca, te = set(map(str, train_ids)), set(map(str, calib_ids)), set(map(str, test_ids))
    assert tr.isdisjoint(ca), f"{len(tr & ca)} training ids leaked into calibration set"
    assert tr.isdisjoint(te), f"{len(tr & te)} training ids leaked into test set"
    assert ca.isdisjoint(te), f"{len(ca & te)} ids shared between calibration and test"
    return True


# --------------------------------------------------------------------------- #
# calibrators
# --------------------------------------------------------------------------- #
class _BaseCalibrator:
    name = "base"
    def fit(self, p, y):  # pragma: no cover - interface
        raise NotImplementedError
    def predict(self, p):  # pragma: no cover - interface
        raise NotImplementedError
    @staticmethod
    def _prep(p):
        return np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)


class Uncalibrated(_BaseCalibrator):
    name = "uncalibrated"
    def fit(self, p, y): return self
    def predict(self, p): return self._prep(p)


class PlattScaling(_BaseCalibrator):
    """Sigmoid recalibration in logit space: p' = sigmoid(A*logit(p) + B)."""
    name = "platt"
    def fit(self, p, y):
        z = logit(self._prep(p)).reshape(-1, 1)
        self._lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._lr.fit(z, np.asarray(y).ravel())
        return self
    def predict(self, p):
        z = logit(self._prep(p)).reshape(-1, 1)
        return self._lr.predict_proba(z)[:, 1]


class IsotonicCalibration(_BaseCalibrator):
    name = "isotonic"
    def fit(self, p, y):
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._iso.fit(self._prep(p), np.asarray(y, dtype=float).ravel())
        return self
    def predict(self, p):
        return np.clip(self._iso.predict(self._prep(p)), 0.0, 1.0)


class TemperatureScaling(_BaseCalibrator):
    """Single scalar T>0 on logits, fit by NLL minimisation: p' = sigmoid(logit(p)/T)."""
    name = "temperature"
    def fit(self, p, y):
        z = logit(self._prep(p)); y = np.asarray(y, dtype=float).ravel()
        def obj(T):
            T = max(T, 1e-3)
            return nll(sigmoid(z / T), y)
        res = minimize_scalar(obj, bounds=(1e-2, 1e2), method="bounded")
        self.T_ = float(max(res.x, 1e-3))
        return self
    def predict(self, p):
        return sigmoid(logit(self._prep(p)) / self.T_)


class BetaCalibration(_BaseCalibrator):
    """
    Beta calibration (Kull et al. 2017): LR on features [ln(p), -ln(1-p)].
    The binary-correct member of the Dirichlet family.
    """
    name = "beta"
    def fit(self, p, y):
        p = np.clip(self._prep(p), _EPS, 1 - _EPS)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        self._lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._lr.fit(X, np.asarray(y).ravel())
        return self
    def predict(self, p):
        p = np.clip(self._prep(p), _EPS, 1 - _EPS)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        return self._lr.predict_proba(X)[:, 1]


class HistogramBinning(_BaseCalibrator):
    """Equal-mass histogram binning: predict the empirical positive rate of the bin."""
    name = "histogram"
    def __init__(self, n_bins=15):
        self.n_bins = n_bins
    def fit(self, p, y):
        p = self._prep(p); y = np.asarray(y, dtype=float).ravel()
        order = np.argsort(p, kind="mergesort")
        self.edges_ = [0.0]
        self.values_ = []
        for b in np.array_split(order, self.n_bins):
            if b.size == 0:
                continue
            self.edges_.append(p[b].max())
            self.values_.append(y[b].mean())
        self.edges_[-1] = 1.0
        self.edges_ = np.array(self.edges_)
        self.values_ = np.array(self.values_)
        return self
    def predict(self, p):
        p = self._prep(p)
        idx = np.clip(np.searchsorted(self.edges_[1:-1], p, side="right"),
                      0, len(self.values_) - 1)
        return self.values_[idx]


class BBQCalibration(_BaseCalibrator):
    """Bayesian Binning into Quantiles (Naeini et al. 2015) via netcal, if installed."""
    name = "bbq"
    def fit(self, p, y):
        try:
            from netcal.binning import BBQ
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "BBQ needs netcal. Install with: pip install netcal  "
                "(the other calibrators run without it)."
            ) from e
        self._m = BBQ()
        self._m.fit(self._prep(p), np.asarray(y, dtype=float).ravel())
        return self
    def predict(self, p):
        return np.clip(np.asarray(self._m.transform(self._prep(p))).ravel(), 0.0, 1.0)


class HybridCalibrator(_BaseCalibrator):
    """
    The locked rule: n_calib < switch_threshold_n -> Platt, else isotonic.
    Records the branch in .method_ so every cell's choice is auditable.
    """
    name = "hybrid"
    def __init__(self, switch_threshold_n=DEFAULT_SWITCH_THRESHOLD_N):
        self.switch_threshold_n = switch_threshold_n
    def fit(self, p, y):
        n = len(np.asarray(y).ravel())
        self.method_ = "platt" if n < self.switch_threshold_n else "isotonic"
        self._inner = PlattScaling() if self.method_ == "platt" else IsotonicCalibration()
        self._inner.fit(p, y)
        self.n_calib_ = n
        return self
    def predict(self, p):
        return self._inner.predict(p)


# --------------------------------------------------------------------------- #
# registry / factory
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "uncalibrated": Uncalibrated,
    "platt": PlattScaling,
    "isotonic": IsotonicCalibration,
    "temperature": TemperatureScaling,
    "beta": BetaCalibration,
    "histogram": HistogramBinning,
    "bbq": BBQCalibration,
    "hybrid": HybridCalibrator,
}
AVAILABLE_METHODS = tuple(_REGISTRY.keys())


def get_calibrator(name, **kwargs):
    """Factory. e.g. get_calibrator('hybrid', switch_threshold_n=1000)."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown calibrator '{name}'. available: {AVAILABLE_METHODS}")
    return _REGISTRY[key](**kwargs)


def fit_predict(name, p_calib, y_calib, p_eval, **kwargs):
    """Convenience: fit a named calibrator on the calibration split, apply to the eval split."""
    cal = get_calibrator(name, **kwargs).fit(p_calib, y_calib)
    out = cal.predict(p_eval)
    return out, cal
