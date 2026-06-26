"""
inference.py — load a DeepfakeBench detector checkpoint and emit per-frame scores
for the CDTS calibration pipeline.

Design:
  - load_detector(): handles the DeepfakeBench import maze (package __init__ pulls
    video-model deps; we import the single detector module in isolation) and the
    build_backbone torch.load(None) issue (we no-op the pretrained load since trained
    weights overwrite it anyway).
  - score_manifest(): runs a loaded model over a manifest (optionally filtered),
    returns a per-frame DataFrame with logits + prob + labels, sliceable by
    method/split/identity for downstream calibration and coupling analysis.

Scores are saved once and calibrated many times (inference is the expensive step;
calibration/metrics operate on the saved score table).

Key facts (confirmed for this repo's DeepfakeBench pin):
  - checkpoint state_dict keys are prefixed 'module.backbone.' -> strip 'module.'
  - forward: model({'image': x}, inference=True) -> {'cls': logits, 'prob': softmax[:,1], 'feat'}
  - preprocessing: RGB, resize 256 BILINEAR, normalize mean=std=0.5  (-> ~[-1,1])
"""
import os, sys, io, shutil, importlib.util, types
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def _ensure_dfb_on_path(dfb_root):
    for p in [f"{dfb_root}/training", dfb_root]:
        if p not in sys.path:
            sys.path.insert(0, p)


# detector module file + class name for each backbone we support
_DETECTOR_SPECS = {
    "xception":       ("xception_detector.py",       "XceptionDetector"),
    "efficientnetb4": ("efficientnetb4_detector.py", "EfficientDetector"),
    "f3net":          ("f3net_detector.py",          "F3netDetector"),
    "clip":           ("clip_detector.py",           "CLIPDetector"),
}


def _import_detector_class(dfb_root, backbone_name):
    """Import a single detector class without triggering detectors/__init__.py
    (which imports the whole video-model stack and its heavy deps)."""
    if backbone_name not in _DETECTOR_SPECS:
        raise ValueError(f"unsupported backbone '{backbone_name}'; add it to _DETECTOR_SPECS")
    fname, cls_name = _DETECTOR_SPECS[backbone_name]
    train_dir = f"{dfb_root}/training"
    det_dir = f"{train_dir}/detectors"

    # stub the 'detectors' package so `from detectors import DETECTOR` resolves
    # without running its __init__ (which pulls slowfast/fvcore/etc.).
    # ALWAYS rebuild against the CURRENT metrics.registry (a stale stub from a
    # previous cell can hold a divergent registry instance -> empty-registry KeyError).
    from metrics.registry import DETECTOR
    stub = types.ModuleType("detectors")
    stub.DETECTOR = DETECTOR
    stub.__path__ = [det_dir]
    sys.modules["detectors"] = stub
    # ensure the networks package backbones are registered (populates BACKBONE);
    # import the specific backbone module the detector needs.
    import importlib as _il
    try:
        _il.import_module("networks")
    except Exception:
        # networks/__init__ may also pull heavy deps; import the xception net directly
        _bspec = importlib.util.spec_from_file_location(
            "networks.xception", f"{train_dir}/networks/xception.py")
        if _bspec is not None:
            _bmod = importlib.util.module_from_spec(_bspec)
            sys.modules.setdefault("networks", types.ModuleType("networks"))
            sys.modules["networks"].__path__ = [f"{train_dir}/networks"]
            sys.modules["networks.xception"] = _bmod
            _bspec.loader.exec_module(_bmod)

    mod_name = f"detectors.{backbone_name}_detector"
    spec = importlib.util.spec_from_file_location(mod_name, f"{det_dir}/{fname}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def load_detector(dfb_root, backbone_name, ckpt_path, device=None, backbone_config=None):
    """Build a DeepfakeBench detector and load trained weights.

    dfb_root: path to external/DeepfakeBench
    backbone_name: 'xception' | 'efficientnetb4' | 'f3net' | 'clip'
    ckpt_path: path to the trained .pth (on Drive or local)
    Returns: (model.eval() on device, device_str)
    """
    _ensure_dfb_on_path(dfb_root)
    DetectorClass = _import_detector_class(dfb_root, backbone_name)

    config = {
        "model_name": backbone_name,
        "backbone_name": backbone_name,
        "backbone_config": backbone_config or {
            "mode": "original", "num_classes": 2, "inc": 3, "dropout": False},
        "pretrained": None,
        "resolution": 256,
        "loss_func": "cross_entropy",
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
    }

    # copy checkpoint to local disk if it's on Drive (Drive FUSE mount has
    # intermittent torch.load seek failures; local read is reliable).
    # Use a path-hash in the local name so DIFFERENT checkpoints of the same
    # backbone (e.g. train_on_fs vs train_on_fr, both 88MB) never collide.
    if ckpt_path.startswith("/content/drive"):
        import hashlib
        tag = hashlib.md5(ckpt_path.encode()).hexdigest()[:8]
        local = f"/content/_ckpt_{backbone_name}_{tag}.pth"
        if not os.path.exists(local) or os.path.getsize(local) != os.path.getsize(ckpt_path):
            shutil.copy(ckpt_path, local)
        ckpt_path = local

    # build model, bypassing build_backbone's torch.load(config['pretrained'])
    # which fails on None (we load full trained weights right after)
    _orig = torch.load
    def _patched(f, *a, **k):
        return {} if f is None else _orig(f, *a, **k)
    torch.load = _patched
    try:
        model = DetectorClass(config)
    finally:
        torch.load = _orig

    # load trained weights, stripping the DataParallel 'module.' prefix
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = {(k[len("module."):] if k.startswith("module.") else k): v
              for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, device, {"missing": len(missing), "unexpected": len(unexpected)}


# ----------------------------------------------------------------------------
# Preprocessing + inference
# ----------------------------------------------------------------------------
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _load_frame(path, res=256):
    img = Image.open(path).convert("RGB").resize((res, res), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


def score_manifest(model, device, manifest, batch_size=64, res=256,
                   limit_per_method=None, methods=None, splits=None, verbose=True):
    """Run model over manifest rows; return a per-frame score DataFrame.

    manifest: DataFrame with columns frame_path,label,video_id,identity_id,method,split
    methods/splits: optional filters (lists)
    limit_per_method: optional cap (for quick runs)
    Returns DataFrame: frame_path,video_id,identity_id,method,split,label,
                       logit_real,logit_fake,prob_fake
    """
    df = manifest
    if methods is not None:
        df = df[df["method"].isin(methods)]
    if splits is not None and "split" in df.columns:
        df = df[df["split"].isin(splits)]
    if limit_per_method is not None:
        df = df.groupby("method", group_keys=False).apply(
            lambda g: g.sample(n=min(limit_per_method, len(g)), random_state=0))
    df = df.reset_index(drop=True)

    rows = []
    n = len(df)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = df.iloc[i:i+batch_size]
            imgs, keep_idx = [], []
            for j, r in chunk.iterrows():
                fp = r["frame_path"]
                if os.path.exists(fp):
                    try:
                        imgs.append(_load_frame(fp, res))
                        keep_idx.append(j)
                    except Exception:
                        pass
            if not imgs:
                continue
            x = torch.stack(imgs).to(device)
            out = model({"image": x}, inference=True)
            logits = out["cls"].detach().cpu().numpy()      # [B,2]
            probs = out["prob"].detach().cpu().numpy()       # [B] fake-class
            for k, j in enumerate(keep_idx):
                r = df.loc[j]
                rows.append({
                    "frame_path": r["frame_path"],
                    "video_id": str(r["video_id"]),
                    "identity_id": str(r.get("identity_id", "")),
                    "method": r["method"],
                    "split": r.get("split", ""),
                    "label": int(r["label"]),
                    "logit_real": float(logits[k, 0]),
                    "logit_fake": float(logits[k, 1]),
                    "prob_fake": float(probs[k]),
                })
            if verbose and (i // batch_size) % 10 == 0:
                print(f"  scored {min(i+batch_size, n)}/{n}")
    return pd.DataFrame(rows)


def quick_auc(scores_df):
    """Sanity AUC from a score DataFrame (overall and per-method)."""
    from sklearn.metrics import roc_auc_score
    out = {}
    y, p = scores_df["label"].values, scores_df["prob_fake"].values
    if len(np.unique(y)) > 1:
        out["overall"] = roc_auc_score(y, p)
    for m, g in scores_df.groupby("method"):
        if g["label"].nunique() > 1:
            out[m] = roc_auc_score(g["label"].values, g["prob_fake"].values)
    return out
