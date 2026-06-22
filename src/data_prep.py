"""
data_prep.py — data layer for the CDTS pipeline.

DeepfakeBench's rearrange.py emits one JSON per dataset describing videos -> frames.
This module turns that JSON into a flat per-frame MANIFEST tagged with:

    frame_path, dataset, label (0 real / 1 fake), video_id, identity_id, era, split_pool

identity_id is the load-bearing column: leakage_safe_split(groups=identity_id) in
calibration.py uses it to keep the SAME person out of both the calibration and test
halves. When the identity is unknown we fall back to video_id (each video its own group),
which can never leak — it only makes the split more conservative.

The manifest builder is deliberately STRUCTURE-AGNOSTIC: it walks the JSON recursively and
treats any node holding a list/dict of image paths as a video, so it does not break if
DeepfakeBench's exact nesting differs across datasets or versions. Tested on a synthetic
DeepfakeBench-style JSON in this module's __main__ self-test.
"""

from __future__ import annotations
import os, json, glob, re
import yaml
import pandas as pd

_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
_REAL_MARKERS = ("real", "original", "youtube", "actors", "celeb-real",
                 "ff-real", "-real", "/real/", "pristine")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_configs(repo):
    """Read paths/datasets/timeline/experiment yaml from <repo>/config."""
    cfg = {}
    for name in ("paths", "datasets", "timeline", "experiment"):
        p = os.path.join(repo, "config", f"{name}.yaml")
        cfg[name] = yaml.safe_load(open(p)) if os.path.exists(p) else {}
    return cfg


def enabled_datasets(datasets_cfg):
    """Flatten the grouped datasets.yaml to {name: meta} for enabled entries only."""
    out = {}
    for group in datasets_cfg.values():
        if not isinstance(group, dict):
            continue
        for name, meta in group.items():
            if isinstance(meta, dict) and meta.get("enabled"):
                out[name] = meta
    return out


def era_lookup(timeline_cfg):
    """method_name -> era string, from timeline.yaml."""
    out = {}
    for era, methods in (timeline_cfg.get("eras") or {}).items():
        for m in methods:
            out[m.lower()] = str(era)
    return out


# --------------------------------------------------------------------------- #
# identity rules (safe default = video is its own group)
# --------------------------------------------------------------------------- #
def identity_from_videoid(dataset, video_id):
    """
    Per-dataset identity extraction. Defaults to the video_id itself (conservative,
    never leaks). Refinements added only where the naming convention is well known.
    """
    d = dataset.lower()
    vid = str(video_id)
    if "faceforensics" in d or d.startswith("ff"):
        # FF++ ids look like '033_097' (target_source); group by target id
        return vid.split("_")[0]
    if "celeb" in d:
        # Celeb-DF ids look like 'id0_0000' or 'id3_id5_0009'; group by first id token
        m = re.match(r"(id\d+)", vid)
        return m.group(1) if m else vid
    if "dfdc" in d:
        # DFDC filenames are opaque hashes; treat each video as its own identity
        return vid
    return vid


# --------------------------------------------------------------------------- #
# structure-agnostic walk of a DeepfakeBench rearrange JSON
# --------------------------------------------------------------------------- #
def _as_frame_list(value):
    """Accept frames as a list of paths or a dict {idx: path}; return list or None."""
    if isinstance(value, list) and value and all(isinstance(s, str) for s in value):
        if any(s.lower().endswith(_IMG_EXT) for s in value):
            return value
    if isinstance(value, dict) and value:
        vals = list(value.values())
        if vals and all(isinstance(s, str) for s in vals) and \
           any(s.lower().endswith(_IMG_EXT) for s in vals):
            return vals
    return None


def _iter_video_nodes(obj, trail=()):
    """
    Yield (frame_paths, trail, node) for every video node found anywhere in the JSON.
    A video node is a dict containing a frames-like value under any key
    ('frames', 'frame_paths', etc.), or a leaf that is itself a frame list.
    """
    if isinstance(obj, dict):
        # does this dict directly hold a frame list?
        for k, v in obj.items():
            fl = _as_frame_list(v)
            if fl is not None:
                yield fl, trail, obj
                break
        else:
            for k, v in obj.items():
                yield from _iter_video_nodes(v, trail + (str(k),))
    elif isinstance(obj, list):
        fl = _as_frame_list(obj)
        if fl is not None:
            yield fl, trail, None


def _infer_label(frame_paths, trail, node):
    """0 real / 1 fake. Prefer an explicit label key, else markers in path/trail."""
    if isinstance(node, dict):
        lab = node.get("label")
        if isinstance(lab, (int, float)):
            return int(lab)
        if isinstance(lab, str):
            return 0 if any(mk in lab.lower() for mk in _REAL_MARKERS) else 1
    hay = (" ".join(trail) + " " + frame_paths[0]).lower()
    return 0 if any(mk in hay for mk in _REAL_MARKERS) else 1


def _video_id(frame_paths, node):
    """Common parent directory of the frames is the video id."""
    if isinstance(node, dict):
        for key in ("video_id", "name", "id"):
            if isinstance(node.get(key), str):
                return node[key]
    parent = os.path.basename(os.path.dirname(frame_paths[0]))
    return parent or os.path.splitext(os.path.basename(frame_paths[0]))[0]


def build_manifest_from_json(dataset, json_path, era_map=None, frames_root=None):
    """
    Expand a DeepfakeBench rearrange JSON into a per-frame manifest DataFrame.
    era_map: optional {method_lower: era}. frames_root: optional prefix to make paths absolute.
    """
    data = json.load(open(json_path))
    era_map = era_map or {}
    rows = []
    for frame_paths, trail, node in _iter_video_nodes(data):
        label = _infer_label(frame_paths, trail, node)
        vid = _video_id(frame_paths, node)
        ident = identity_from_videoid(dataset, vid)
        method = trail[-1].lower() if trail else dataset.lower()
        era = era_map.get(method, "")
        for fp in frame_paths:
            path = os.path.join(frames_root, fp) if frames_root else fp
            rows.append({
                "frame_path": path, "dataset": dataset, "label": int(label),
                "video_id": str(vid), "identity_id": str(ident),
                "method": method, "era": era, "split_pool": "heldout",
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("frame_path").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# DeepfakeBench preprocessing invocation + access/status tracking
# --------------------------------------------------------------------------- #
def dfb_json_path(dfb_root, dataset):
    """Where rearrange.py writes a dataset's JSON (best-effort name match)."""
    folder = os.path.join(dfb_root, "preprocessing", "dataset_json")
    cand = os.path.join(folder, f"{dataset}.json")
    if os.path.exists(cand):
        return cand
    hits = glob.glob(os.path.join(folder, "*.json"))
    norm = dataset.lower().replace("-", "").replace("_", "").replace("+", "")
    for h in hits:
        if norm in os.path.basename(h).lower().replace("-", "").replace("_", "").replace("+", ""):
            return h
    return None


def run_dfb_preprocess(dfb_root):
    """
    Shell out to DeepfakeBench's documented two-step flow. DeepfakeBench is config-driven:
    its own preprocessing/config.yaml must point at the placed raw dataset first (per their
    README). Returns the shell return codes.
    """
    import subprocess
    c1 = subprocess.run(f'cd "{dfb_root}/preprocessing" && python preprocess.py',
                        shell=True)
    c2 = subprocess.run(f'cd "{dfb_root}" && python preprocessing/rearrange.py',
                        shell=True)
    return c1.returncode, c2.returncode


def status_table(repo):
    """One row per enabled dataset: access type + whether DeepfakeBench has processed it."""
    cfg = load_configs(repo)
    dfb = cfg["paths"].get("deepfakebench", os.path.join(repo, "external/DeepfakeBench"))
    rows = []
    for name, meta in enabled_datasets(cfg["datasets"]).items():
        jp = dfb_json_path(dfb, name)
        rows.append({
            "dataset": name,
            "license": meta.get("license", "?"),
            "modality": meta.get("modality", "?"),
            "demo_labels": meta.get("demo_labels", "none"),
            "processed": jp is not None,
            "json": os.path.basename(jp) if jp else "-- pending preprocess --",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # synthetic DeepfakeBench-style JSON: FF++ with a real and a fake method, nested by split
    synth = {
        "FaceForensics++": {
            "FF-real": {"train": {
                "033_097": {"label": "FF-real",
                            "frames": ["ff/real/033_097/0.png", "ff/real/033_097/1.png"]},
                "035_036": {"label": "FF-real",
                            "frames": ["ff/real/035_036/0.png"]},
            }},
            "FF-FS": {"train": {
                "033_097": {"label": "FF-FS",
                            "frames": {"0": "ff/fs/033_097/0.png", "1": "ff/fs/033_097/1.png"}},
            }},
        }
    }
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "_ff_synth.json")
    json.dump(synth, open(p, "w"))
    df = build_manifest_from_json("FaceForensics++", p)
    assert len(df) == 5, f"expected 5 frames, got {len(df)}"
    assert set(df["label"]) == {0, 1}, "label inference failed"
    assert df[df.method == "ff-real"]["label"].eq(0).all(), "real mislabeled"
    assert df[df.method == "ff-fs"]["label"].eq(1).all(), "fake mislabeled"
    # FF++ identity = target id before underscore -> '033' groups real+fake of 033_097
    assert set(df[df.video_id == "033_097"]["identity_id"]) == {"033"}, "identity rule failed"
    # the same identity appears in BOTH a real and a fake method -> group split will hold it out
    ident_033 = df[df.identity_id == "033"]
    assert set(ident_033["label"]) == {0, 1}, "identity should span real+fake"
    print("data_prep self-test PASSED")
    print(df.to_string(index=False))
