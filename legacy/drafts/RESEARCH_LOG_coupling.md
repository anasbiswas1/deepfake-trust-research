# CDTS Research Log — Competence–Calibration Coupling

**Project:** Calibrated Deepfake Trust Score (CDTS) — reframing deepfake detection as a
calibrated self-auditing trust instrument.
**Repo:** github.com/anasbiswas1/deepfake-trust-research
**Phase:** NB02 (FF++ calibration) → NB03 (DF40 scaling) → NB04 (confound-breaker)
**Last updated:** end of NB04 confound-breaker session
**Latest commit:** 647e796

---

## TL;DR — the finding is ROBUST (confound resolved)

**Competence–Calibration Coupling:** calibration error (ECE after calibration) **rises as
detection competence (AUC) falls.** Now demonstrated:
- Across **20 points**, 2 datasets, 3 checkpoint conditions, 4+ forgery families: **r = -0.80, p < 0.0001**
- **Within a single dataset** (DF40 alone, 12 pts): **r = -0.94, p < 0.0001**
- **Causally** — deliberately degrading competence (mismatched checkpoint) reliably degrades
  calibration: 4/4 forced-low-competence DF40 points showed high ECE_cal.

**The dataset confound (raised in NB03) is BROKEN.** It's competence, not dataset.

---

## The pipeline (validated, reusable)

- **Inference:** src/inference.py — loads DeepfakeBench detectors, scores manifests.
  Handles: import-isolation (bypasses video-model dep stack), torch.load(None) in
  build_backbone, module.backbone. key stripping, local-disk checkpoint copy with
  **path-hash naming** (different same-size checkpoints don't collide — this bug bit us:
  train_on_fs vs train_on_fr xception both 88MB silently reused the same local copy until fixed),
  BACKBONE registry population on fresh runtimes.
- **Calibration:** src/calibrate_scores.py + src/calibration.py (HybridCalibrator =
  Option D: Platt n<1000 / isotonic n>=1000) + src/metrics.py (ECE equal-mass 15-bin,
  Brier, bootstrap CI). Leakage-safe split by identity_id (group-disjoint).
- **metrics name collision:** your src/metrics.py vs DeepfakeBench's training/metrics/
  package need OPPOSITE path orders. Inference -> DFB metrics first; calibration -> src first.
  Switch by purging sys.modules + reordering path between phases.
- **Score -> save -> calibrate-many-times.** Inference is the GPU-expensive step.

---

## RESULT 1 — FF++ calibration (NB02), commit add3cd4

Xception, FF++ c23 test (22,388 frames), two checkpoints.

**Pooled:** train_on_fs ECE 0.455->0.047 (89.6% reduction); train_on_fr 0.694->0.207 (70.2%).
(FR less so — broadly incompetent on FF++, all AUC <=0.66. Can't calibrate away near-randomness.)

**Per-method AUC:** FS-trained -> faceswap 0.97, deepfakes 0.81, face2face 0.63, neuraltextures 0.60.
FR-trained -> faceswap 0.39(!), deepfakes 0.66, face2face 0.49, neuraltextures 0.64.

**Coupling (8 cells):** Pearson r=-0.82 (p=0.013), Spearman rho=-0.81 (p=0.015).
ECE_cal spans 0.036-0.343, AUC 0.39-0.97, moving together.

**Mechanism:** a calibrator learns the score<->correctness mapping in the detector's competent
regime. Applied where the detector is incompetent (low AUC), the mapping fails -> trust scores
unreliable exactly where you'd most need them to say "don't trust me."

---

## RESULT 2 — DF40 scaling (NB03), commit 0e567c7

FS Xception on 8 DF40 generators (cdf source), paired with cropped Celeb-DF reals.

**Per-method:** StyleGAN2 0.65/0.063, sd2.1 0.69/0.044, ddim 0.77/0.041, fomm 0.80/0.054,
inswap 0.82/0.073, facevid2vid 0.82/0.064, blendface 0.94/0.022, simswap 0.95/0.024 (AUC/ECE_cal).

**DF40-only coupling:** r=-0.585 (p=0.13, n.s.). DF40 calibrated WELL across its range —
ECE_cal 0.02-0.07 regardless of AUC. The clean FF++ coupling did NOT replicate within DF40.

**Combined 16 pts (FF++ + DF40):** r=-0.779 (p=0.0004). Significant, BUT carried by the FF++
spread. The figure (figures/coupling_competence_calibration.png) showed DF40 clustered flat
(low ECE) while FF++ spanned the range -> raised the **confound: competence or dataset?**

---

## RESULT 3 — CONFOUND-BREAKER (NB04), commit 647e796 -- KEYSTONE

**Question:** is the coupling competence-driven, or a dataset effect (FF++ harder to calibrate)?

**Method:** force LOW competence on DF40 by scoring with a MISMATCHED checkpoint —
train_on_fr (reenactment-trained) on DF40's FS/EFS methods (simswap, blendface, StyleGAN2,
sd2.1). Reenactment-trained model on swap/synthesis -> AUC craters.

**Result — all 4 landed top-left (low AUC, HIGH ECE_cal):**
| method | AUC | ECE_cal |
|---|---|---|
| StyleGAN2 (FR) | 0.256 | 0.268 |
| sd2.1 (FR) | 0.329 | 0.268 |
| simswap (FR) | 0.363 | 0.301 |
| blendface (FR) | 0.457 | 0.248 |

**4/4 low-AUC DF40 points show HIGH ECE_cal.** When DF40 is forced into low competence, its
calibration fails exactly like FF++. The earlier DF40 flatness was because the FS checkpoint
kept those methods at moderate-high competence — NOT because DF40 is intrinsically easy to calibrate.

- **Within-DF40 (12 pts, FS + FR-mismatch): r = -0.939, p < 0.0001** (was -0.585 n.s.).
- **Full (20 pts): r = -0.802, p < 0.0001.**

**VERDICT: confound broken. Coupling is competence-driven, not dataset.** Demonstrated within
a single dataset and causally (competence manipulation -> calibration failure).

Figure: figures/coupling_with_FRmismatch.png (green = FR-mismatch DF40 points filling top-left).

---

## CONTRIBUTION (final defensible form)

Calibration of deepfake trust scores is coupled to detection competence (r ~ -0.8 to -0.94,
demonstrated within and across datasets and causally via competence manipulation). Modern
calibrators succeed when the detector is competent but fail when competence collapses —
regardless of dataset, forgery family, or calibration-data volume. Since competence is lowest
precisely for novel, unseen generators, static calibration is unreliable exactly where
deployment risk is highest. This motivates competence-aware trust scoring (CDTS).

The arc — find (FF++) -> stress-test (DF40, nearly broke it) -> diagnose confound -> resolve with
a designed experiment (NB04) — is the empirical core of the paper. Done.

---

## NEXT STEPS (priority order)

1. **NB05 — third backbone (DOING NEXT):** EfficientNetB4 or F3Net (both in weights/train_on_fs/)
   on FF++ + a few DF40 methods -> prove the coupling is NOT Xception-specific. If a different
   architecture shows the same coupling, it's architecture-general (strong). Reuses everything;
   inference.py already supports efficientnetb4 + f3net in _DETECTOR_SPECS.
2. **Scale to more DF40 methods** (30+ beyond the 9 core; train+test zips via shortcuts) for
   denser generator coverage along the AUC axis.
3. **Start drafting** — the central figure + result are done; the paper is buildable.
4. **Optional robustness:** vary calibration-data fraction (small-calib test) to show the
   coupling sharpens under data scarcity (realistic deployment condition).

---

## WORKFLOW NOTES (don't relearn)

- Checkpoints/frames load reliably only from LOCAL /content/ disk, not Drive FUSE (torch.load
  seek failures). inference.py copies checkpoints local automatically (with path-hash naming).
- NEVER git add -A (hangs on large data folders) — stage explicit files.
- NEVER flush_and_unmount (corrupts mount, errno 107) — only force_remount.
- Git identity dies on runtime restart — restore .gitconfig/.git-credentials from
  Drive CDTS_Research/ at session start (the cp-from-Drive cell at top of each notebook).
- DF40 via Drive shortcuts (throttle-free); 9 core methods in data/df40_core/.
- DF40 method JSONs come in _cdf (Celeb-DF source) and _ff (FF++ source) variants — match the
  variant to the zip you have. cdf zips pair with cropped Celeb-DF reals; ff with ffpp_real.zip.
- Colab upload to Drive lags (sync delay) — os.path.exists may show False then True; import
  modules via importlib.util.spec_from_file_location when normal import fails (FUSE stat caching).
- Inference env vs calibration env: purge sys.modules + reorder sys.path between them
  (metrics collision). Already-loaded model keeps working after the switch (path only matters
  for importing, not running).
- Two-pass loop for many methods: score all (inference env) then calibrate all (calibration env)
  — avoids thrashing the path between every method.
