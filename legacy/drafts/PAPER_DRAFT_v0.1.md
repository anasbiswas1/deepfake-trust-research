# Competence Is the Hidden Variable: Calibration of Deepfake Trust Scores Degrades With Detection Competence

*(working title — alternatives: "Calibration You Can't Trust: Competence-Conditional Reliability of Deepfake Detectors"; "When Confidence Lies: Competence-Coupled Miscalibration in Deepfake Detection")*

**Author:** Md Anas Biswas (+ supervisor TBD)
**Status:** DRAFT v0.1 — scaffold with validated results slotted in
**Repo:** github.com/anasbiswas1/deepfake-trust-research

---

## ABSTRACT (draft — ~200 words, write last, placeholder now)

Deepfake detectors are increasingly deployed as *trust signals*: a detector's confidence is
taken to indicate how much a flagged (or cleared) media item should be trusted. This places
weight on the detector's **calibration** — whether its stated probabilities match empirical
correctness — yet calibration is almost never evaluated as detection difficulty varies across
the rapidly expanding landscape of generation methods. We show that the calibration quality of
deepfake trust scores is **coupled to detection competence**: post-hoc calibration error (ECE)
rises sharply as detection performance (AUC) falls. Across [N=24] (detector, generator) cells
spanning two backbone architectures (Xception, EfficientNet-B4), two source datasets
(FaceForensics++, Celeb-DF/DF40), and four forgery families (face-swap, reenactment,
entire-face-synthesis), we find a strong negative association (Pearson r = -0.81, p < 1e-5).
The coupling holds *within* a single dataset (r = -0.94) and is **causal**: deliberately
degrading competence via a deliberately mismatched detector reliably degrades calibration
(4/4 forced-low-competence cells). Because competence is lowest precisely for novel, unseen
generators — the highest-risk deployment scenario — a single static calibration is unreliable
exactly where reliability matters most. We argue for **competence-aware trust scoring**.

---

## 1. INTRODUCTION

**Paragraph 1 — the deployment reality.** Deepfake detectors are used as trust signals in
content moderation, journalism verification, and platform integrity. Decisions increasingly
hinge not just on the binary real/fake call but on the detector's *confidence* — its
probability is read as a degree of trust. [cite deployment/usage].

**Paragraph 2 — the gap.** A confidence is only meaningful if it is *calibrated*: among items
the detector calls 90% likely fake, ~90% should be fake. Calibration of classifiers is
well-studied [Guo et al. 2017; Platt 1999; Zadrozny & Elkan], but for deepfake detection it is
typically reported, if at all, as a single number on a single benchmark. The generator
landscape, meanwhile, is exploding — face-swap, reenactment, GAN synthesis, diffusion — and a
detector's competence varies enormously across it. **Whether calibration survives this
competence variation is unexamined.**

**Paragraph 3 — our question and finding.** We ask: does the calibration of a deepfake trust
score depend on how well the detector actually detects? We find a strong, robust coupling —
calibration error rises as competence falls — that holds across architectures, datasets, and
forgery families, holds within a single dataset, and is causal under competence manipulation.

**Paragraph 4 — implication & contribution.** [State CDTS motivation + list contributions:]
- (C1) We document a **competence–calibration coupling** in deepfake trust scores
  (r = -0.8 to -0.94), the first systematic characterization to our knowledge.
- (C2) We establish it is **not an artifact** of dataset or architecture via a within-dataset
  analysis and a controlled competence-manipulation experiment.
- (C3) We draw the deployment implication: static calibration fails for novel generators, and
  motivate **competence-aware trust scoring (CDTS)**.

---

## 2. RELATED WORK

**2.1 Deepfake detection & benchmarks.** FaceForensics++ [Rossler 2019], Celeb-DF [Li 2020],
DF40 [Yan 2024] and the DeepfakeBench framework [Yan 2023]. Generation-family taxonomy
(FS/FR/EFS/FE). [Position: prior work measures AUC across generators; we add the trust layer.]

**2.2 Calibration of neural classifiers.** Temperature scaling [Guo 2017], Platt scaling,
isotonic regression [Zadrozny & Elkan], beta calibration, histogram binning; ECE and its
debiased variants. [Position: calibration studied for standard classifiers, rarely under
distribution shift, almost never for deepfake detection across generators.]

**2.3 Calibration under distribution shift.** [Ovadia et al. 2019 — calibration degrades
under shift.] [Position: our generators ARE the shift axis; we connect shift-degradation
specifically to detection competence and show the relationship is lawful.]

**2.4 Trust / selective prediction in forensics.** [Any work on abstention, uncertainty for
deepfakes.] [Position: motivates competence-aware scoring as the response to our finding.]

---

## 3. METHOD

**3.1 Detectors & checkpoints.** Xception and EfficientNet-B4 from DeepfakeBench, trained on
the FS (face-swap) family. A reenactment-trained (FR) checkpoint used for the
competence-manipulation experiment (Sec 5.3). [Table: backbones, training family, params.]

**3.2 Data & manifests.** FF++ c23 (159,627 frames cropped; official 720/140/140 video split,
test used here = 22,388 frames). Celeb-DF v2 test split (518 videos, 16,420 frames). DF40
generators (cdf source), paired with Celeb-DF reals. Leakage-safe handling: identity-disjoint
splits throughout. [Table: datasets, #frames, #identities, real/fake balance.]

**3.3 Trust score & calibration.** The detector's fake-class probability is the raw trust
score. Post-hoc calibration via a hybrid scheme (Platt scaling for small calibration sets
n<1000; isotonic regression for n>=1000). Calibration fit on an identity-disjoint split,
evaluated on held-out identities. [This is your Option D / HybridCalibrator.]

**3.4 Metrics.** AUC (competence); Expected Calibration Error (ECE, 15-bin equal-mass) before
and after calibration (ECE_raw, ECE_cal); Brier score. Bootstrap CIs (B=2000, identity- and
label-stratified). [Define ECE formula.]

**3.5 The coupling measurement.** Each (detector, generator) pair yields one (AUC, ECE_cal)
point. We measure the association across all such points (Pearson, Spearman).

---

## 4. CALIBRATION IMPROVES TRUST SCORES (baseline)

[Result 1 pooled — establishes calibration helps, sets up the nuance.]

Raw detectors are substantially miscalibrated; post-hoc calibration reduces ECE markedly:
- Xception/FS on FF++: ECE 0.455 -> 0.047 (**89.6% reduction**).
- But for a broadly-incompetent detector (FR on FF++, all AUC <= 0.66): ECE 0.694 -> 0.207
  (70.2%) — calibration helps less, because **you cannot calibrate away near-randomness.**

[Table 1: per-checkpoint pooled raw/calibrated ECE + Brier, with CIs.]
[This already foreshadows the coupling: calibration's success depends on competence.]

---

## 5. THE COMPETENCE–CALIBRATION COUPLING (core result)

**5.1 Within FF++ (cross-forgery).** Two checkpoints x four FF++ methods = 8 cells. Calibration
error tracks competence: **Pearson r = -0.82 (p = 0.013), Spearman rho = -0.81 (p = 0.015).**
AUC spans 0.39-0.97, ECE_cal spans 0.036-0.343. [Table 2.]

**5.2 Across the DF40 generator landscape.** FS-Xception on 8 DF40 generators (FS/FR/EFS
families). DF40 alone shows a WEAKER association (r = -0.585, n.s.) — calibration succeeds
across DF40's moderate-to-high competence range. Combined with FF++ (16 pts): r = -0.78
(p = 4e-4). [Fig 1: scatter colored by dataset — reveals DF40 cluster vs FF++ spread.]
[Honest framing: this raised the question — competence or dataset?]

**5.3 Breaking the confound: competence manipulation (the key experiment).** To test whether
the coupling is competence-driven or a dataset artifact, we *force* low competence on DF40 by
scoring with a deliberately mismatched (reenactment-trained) detector on swap/synthesis
generators. This craters AUC (0.26-0.46). **All 4/4 forced-low-competence DF40 cells show high
ECE_cal (0.25-0.30)** — landing exactly with the FF++ low-competence points.
- **Within-DF40 now (12 cells): r = -0.94 (p < 1e-4)** — the coupling that was absent appears
  once competence is varied within the *same dataset*. The dataset confound is ruled out.
[Fig 2: scatter with FR-mismatch points filling the low-AUC/high-ECE region.]

**5.4 Architecture-generality.** EfficientNet-B4 (a different architecture) shows the same
coupling on FF++: **r = -0.95 (p = 0.048)**, points falling on the Xception trend.
**Combined across everything (24 cells, 2 architectures, 2 datasets, 4 forgery families):
Pearson r = -0.81, p = 1.5e-6.** [Fig 3: scatter colored by architecture — the headline figure.]

**5.5 Mechanism.** A calibrator learns the score<->correctness mapping in the detector's
*competent* regime. Applied where the detector is incompetent (low AUC), the mapping no longer
holds: the score distribution and its relationship to correctness have changed. Calibration
repairs miscalibration but cannot create the competence on which a reliable mapping depends.
Trust scores are therefore least reliable exactly where competence is lowest.

---

## 6. DISCUSSION & IMPLICATIONS

**6.1 Deployment risk is concentrated where calibration fails.** Competence is lowest for
novel, unseen generators — precisely the case a deployed system most needs to handle and most
needs honest uncertainty for. A single static calibration gives a false sense of reliability
on exactly these inputs.

**6.2 Toward competence-aware trust scoring (CDTS).** [Sketch the proposal your program builds
toward: estimate competence at inference (or per-generator-family), condition the trust score /
abstention threshold on it. This paper motivates it; full CDTS is future/companion work.]

**6.3 Limitations.** [Honest: AUC-competence is per-generator (needs generator labels or
estimation in the wild); ECE has known estimator sensitivity (we use equal-mass + report
debiased in appendix); two architectures, two source datasets — broad but not exhaustive;
image-level (not video-level temporal) detectors.]

---

## 7. CONCLUSION

Calibration of deepfake trust scores is coupled to detection competence — robustly, within and
across datasets and architectures, and causally. Confidence from a deepfake detector cannot be
trusted uniformly; it is least trustworthy where the detector is weakest, which is where it
matters most. Trust scoring must be made competence-aware.

---

## FIGURES & TABLES (have / need)

- **Fig 1** (HAVE): coupling_competence_calibration.png — 16pts, dataset-colored.
- **Fig 2** (HAVE): coupling_with_FRmismatch.png — FR-mismatch points break the confound.
- **Fig 3 / headline** (HAVE): coupling_two_backbones.png — 24pts, architecture-colored.
  [May merge Fig 1-3 into one multi-panel for space; or Fig 3 alone as the headline.]
- **Table 1** (HAVE data): pooled raw/cal ECE+Brier per checkpoint. -> reports/calibration/*pooled*.csv
- **Table 2** (HAVE data): per-(detector,generator) AUC + ECE_raw + ECE_cal, all 24 cells.
  -> coupling_all_16pts.csv + coupling_df40_FRmismatch.csv + coupling_effnetb4_ffpp.csv
- **Table 3** (NEED): summary of correlations per slice (FF++ only, DF40+mismatch, EffNet,
  combined) with r, p, n. [Quick to assemble from saved CSVs.]
- **Mechanism diagram** (NEED, optional): schematic of why calibration fails out-of-competence.

---

## EXPERIMENTS TO STRENGTHEN (the "more experiments" phase, after drafting)

Ranked by value to the draft once it exists:
1. **Denser DF40 coverage** — score the remaining ~30 DF40 generators with FS-Xception. Turns
   "8 generators" into "~40 generators"; makes Fig 3 far more compelling and the claim broader.
   [Reuses NB03/NB05 two-pass flow exactly; data via shortcuts.]
2. **Assemble Table 3** (correlation-per-slice summary) — trivial, from saved CSVs.
3. **EfficientNet-B4 on DF40** — currently EffNet is FF++-only (4 pts). Adding EffNet-on-DF40
   gives a full architecture x dataset grid. Strengthens architecture-generality claim.
4. **Small-calibration-fraction robustness** — refit with tiny calib sets; show ECE_cal spreads
   and coupling sharpens under scarcity (realistic deployment, mechanistic insight).
5. **F3Net (third architecture)** — weights/train_on_fs/f3net.pth already present; inference.py
   supports it. One more architecture point if a reviewer wants >2.
6. **ECE estimator robustness** (appendix) — repeat key correlation with debiased ECE / varying
   bins to show it's not a binning artifact.

---

## WRITING NOTES

- Lead with the deployment framing (trust signals), not "we calibrated detectors" — the
  contribution is conceptual (competence-conditional reliability), not a calibration benchmark.
- The CAUSAL experiment (5.3) is the rhetorical centerpiece — it's what makes this more than a
  correlation. Foreground it.
- Be scrupulously honest about the DF40-alone weakness (5.2) BEFORE resolving it (5.3). The
  honest arc is more convincing than hiding the wrinkle, and pre-empts the obvious reviewer
  objection.
- Keep the CDTS proposal (6.2) appropriately scoped — this paper *motivates* it; don't
  over-claim a full system you haven't built.
