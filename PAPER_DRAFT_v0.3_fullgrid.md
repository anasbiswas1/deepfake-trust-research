# Competence Is the Hidden Variable: Calibration of Deepfake Trust Scores Degrades With Detection Competence

**Md Anas Biswas**¹ · *Supervisor TBD*¹
¹ School of Computing, University of Portsmouth

**Draft v0.3 (32-point grid)** — github.com/anasbiswas1/deepfake-trust-research

---

## Abstract

Deepfake detectors are increasingly deployed as *trust signals*: a detector's confidence that
an item is manipulated is consumed downstream as a degree of trust, informing moderation,
verification, and platform-integrity decisions. This places weight on the detector's
**calibration** — whether its stated probabilities match empirical correctness — yet
calibration is almost never evaluated as detection difficulty varies across the rapidly
expanding landscape of generation methods. We show that the calibration quality of deepfake
trust scores is **coupled to detection competence**: post-hoc calibration error (Expected
Calibration Error, ECE) rises sharply as discriminative performance (AUC) falls. Across 32
(detector, generator) configurations spanning two backbone architectures (Xception,
EfficientNet-B4), two source datasets (FaceForensics++, Celeb-DF/DF40), and four forgery
families (face-swap, reenactment, entire-face-synthesis), we observe a strong negative
association (Pearson r = −0.81, p < 10⁻⁵). The coupling holds *within* a single dataset
(r = −0.94) and is **causal**: deliberately degrading competence by applying a mismatched
detector reliably degrades calibration (4 of 4 forced-low-competence configurations show high
calibration error). Because competence is lowest precisely for novel, unseen generators — the
highest-risk deployment scenario — a single static calibration is unreliable exactly where
reliability matters most. We argue that deepfake trust scoring must be made competence-aware.

---

## 1. Introduction

Deepfake detectors no longer function only as binary classifiers. In content moderation,
journalistic verification, and platform-integrity pipelines, a detector's *confidence* is
read as a degree of trust: an item flagged as fake with 95% probability is treated differently
from one flagged at 55%, and downstream actions — automated removal, human review, public
labelling — are increasingly conditioned on that probability rather than the bare decision.

A probability is only meaningful, however, if it is *calibrated*. Among the items a detector
calls 90% likely to be fake, roughly 90% should in fact be fake; otherwise the confidence
misleads exactly the decisions it is meant to inform. Calibration of neural classifiers is
well studied — modern networks are known to be systematically overconfident, and a family of
post-hoc methods (temperature and Platt scaling, isotonic regression, and others) can largely
correct this on in-distribution data. For deepfake detection, however, calibration is
typically reported, if at all, as a single number on a single benchmark.

This omission is consequential because the generation landscape is not static. New
face-swapping, reenactment, GAN, and diffusion methods appear continually, and a detector's
*competence* — how well it actually discriminates real from fake — varies enormously across
them. A detector trained on face-swap forgeries may be near-perfect on swaps and near-random
on a novel diffusion generator. Whether a detector's *calibration* survives this competence
variation, and what happens to the trustworthiness of its confidence when competence
collapses, has not been systematically examined.

We ask a direct question: **does the calibration of a deepfake trust score depend on how well
the detector actually detects?** We find that it does, strongly and lawfully. Calibration
error rises as detection competence falls — a relationship we term the **competence–calibration
coupling**. We establish that this coupling is not an artifact of any single dataset or
architecture, that it holds *within* a single dataset, and that it is *causal*: when we
deliberately reduce a detector's competence by applying it outside its training family,
calibration degrades in lockstep.

The implication for deployment is sharp. Competence is lowest precisely for the generators a
detector has not seen — the novel forgeries a deployed system most needs to handle correctly,
and for which honest uncertainty is most valuable. A single static calibration, fit once and
applied uniformly, gives a false sense of reliability on exactly these inputs. This motivates
**competence-aware trust scoring**, in which the reliability of the trust score is conditioned
on an estimate of the detector's competence for the input at hand.

Our contributions are:

1. We document a **competence–calibration coupling** in deepfake trust scores — calibration
   error rises as detection competence falls (Pearson r between −0.77 and −0.94 across slices; r = −0.81, p = 2.5×10⁻⁸ pooled over 32 configurations)
   — to our knowledge the first systematic characterization of this relationship.
2. We show the coupling is **not an artifact** of dataset or architecture: it holds within a
   single dataset, replicates across two backbone architectures, and — via a controlled
   competence-manipulation experiment — is demonstrated to be **causal** rather than merely
   correlational.
3. We draw the deployment implication: static calibration is least reliable for novel,
   poorly-detected generators, and we argue for **competence-aware trust scoring**.

---

## 2. Related Work

**Deepfake detection and benchmarks.** Detection has been driven by large benchmarks:
FaceForensics++ (FF++) established a standard suite of four manipulation methods across
face-swap and reenactment families; Celeb-DF improved realism and is widely used as a
cross-dataset test; and DF40 substantially broadened the generator landscape, providing forty
modern generation methods organized into face-swapping, reenactment, entire-face-synthesis,
and face-editing families. The DeepfakeBench framework consolidates detectors, data
preprocessing, and pretrained weights for reproducible evaluation. This line of work measures
detector *competence* (typically AUC) across generators; we add the orthogonal *trust* layer,
asking whether the detector's confidence is calibrated as competence varies.

**Calibration of neural classifiers.** Guo et al. showed modern networks are systematically
overconfident and that simple temperature scaling restores calibration on in-distribution
data; Platt scaling and isotonic regression are long-standing alternatives, and a range of
binning- and Bayesian-based methods exist. Calibration is commonly summarized by the Expected
Calibration Error and its debiased variants. These methods and metrics are studied primarily
for standard classifiers on a fixed distribution; their behaviour as the test distribution
shifts away from training is far less characterized, and for deepfake detection across
generators it is essentially unstudied.

**Calibration under distribution shift.** Work on uncertainty under dataset shift has shown
that calibration generally degrades as test data moves away from the training distribution,
and that the degradation is uneven across methods. Our setting makes the generation method the
explicit shift axis. Rather than treating shift as a monolithic degradation, we connect
calibration error specifically to *detection competence* and show the relationship is strong,
monotone, and causal — a more actionable characterization than "shift hurts calibration."

**Trust, abstention, and selective prediction in forensics.** A complementary response to
detector unreliability is to abstain or defer when uncertain. Our finding motivates such
mechanisms but locates the problem precisely: it is not uncertainty in general but
*competence-conditioned miscalibration* that undermines trust scores, and the response should
condition on competence rather than on raw confidence alone.

---

## 3. Method

We frame deepfake detection as a *trust-scoring* problem: a detector outputs a probability
that an input is manipulated, and this probability is consumed downstream as a degree of
trust. Our study measures how the *reliability* of that probability — its calibration —
relates to the detector's *competence* — its discriminative performance — as the underlying
generation method varies.

### 3.1 Detectors and checkpoints

We use detectors from the DeepfakeBench framework, which provides unified implementations and
pretrained weights for a range of backbones. Our primary backbone is **Xception**, the
canonical and most-studied deepfake-detection architecture; to test architecture-generality we
additionally use **EfficientNet-B4**. Both are binary (real vs. manipulated) classifiers
operating on cropped face images.

The detectors are trained on the **face-swapping (FS)** family of the DF40 benchmark. This
fixes a single, well-defined training distribution and lets generation-method variation at
test time act as a controlled axis of distribution shift. For the competence-manipulation
experiment (Section 5.3) we additionally employ a **reenactment-trained (FR)** Xception
checkpoint, used deliberately outside its training family to induce low competence.

### 3.2 Data and manifests

We evaluate on three sources spanning distinct identities, source datasets, and generation
families. From **FaceForensics++** (c23 compression) we extract and face-crop all frames
(159,627 frames over 1,000 identities) and use the official video-level test split (22,388
frames); FF++ provides four manipulation methods across two families — FaceSwap and Deepfakes
(face-swap), Face2Face and NeuralTextures (reenactment). From **Celeb-DF v2** we crop the
official test split (518 videos, 16,420 frames), whose real frames we use to pair with DF40
fakes. From **DF40** (Celeb-DF-sourced variant) we take pre-cropped fakes for modern
generators spanning face-swap, reenactment, and entire-face-synthesis families; we use eight
core generators in the main analysis (simswap, inswap, blendface; fomm, facevid2vid;
StyleGAN2, sd2.1, ddim), each paired with the Celeb-DF real frames to form a balanced task.

**Leakage control.** All calibration/evaluation splits are *identity-disjoint*: no subject
identity appears in both the calibration-fitting and evaluation partitions. We implement this
with a group-disjoint splitter keyed on identity, with hard assertions that neither identity
nor frame crosses the split. This prevents the optimistic bias that frame-level splitting
would introduce when multiple frames of one video or person appear on both sides.

### 3.3 Trust score and calibration

The raw **trust score** is the detector's softmax probability for the manipulated class. Since
raw neural-network probabilities are typically miscalibrated, we apply post-hoc calibration
learned on a held-out, identity-disjoint calibration partition and evaluated on the remaining
identities. We use a **hybrid calibrator** that adapts to calibration-set size: **Platt
scaling** (a two-parameter logistic fit) when the calibration set is small (n < 1000), and
**isotonic regression** (a flexible non-parametric monotone fit) when larger (n ≥ 1000). This
avoids isotonic regression's tendency to overfit on small samples while retaining its
flexibility where data is sufficient.

### 3.4 Metrics

We quantify two properties of each (detector, generator) configuration. **Competence** is the
area under the ROC curve (**AUC**) of the raw trust scores on the evaluation partition — a
threshold-independent measure of discriminative ability. **Calibration error** is the
**Expected Calibration Error (ECE)** on the evaluation partition, using 15 equal-mass bins
(equal samples per bin, more stable than equal-width bins under skewed score distributions);
we report ECE on raw scores (ECE_raw) and calibrated scores (ECE_cal), and the **Brier score**
for pooled analyses. ECE estimates the gap between confidence and accuracy: binning predictions
by predicted probability, it is the sample-weighted mean absolute difference between each bin's
mean confidence and its empirical accuracy. For pooled results we report 95% bootstrap
confidence intervals (B = 2000, label-stratified) on ECE.

### 3.5 The coupling measurement

Each (detector, generator) pair yields a single point: its competence (AUC) and its
post-calibration error (ECE_cal). The central object of study is the association between these
across all points, measured with both **Pearson** (linear) and **Spearman** (monotone) rank
correlation, reporting coefficient, p-value, and n per slice of the data and combined. A strong
negative association indicates the competence–calibration coupling.

---

## 4. Calibration Improves Trust Scores (Baseline)

We first confirm that calibration is worth doing at all, and in doing so surface the nuance
that motivates the rest of the paper. Raw detectors are substantially miscalibrated, and
post-hoc calibration reduces ECE markedly. For the FS-trained Xception on FF++, pooled ECE
falls from 0.455 to 0.047 — an 89.6% reduction. The benefit, however, is not uniform: for a
*broadly incompetent* detector (the FR checkpoint applied to FF++, where every method's AUC is
at most 0.66), calibration reduces ECE only from 0.694 to 0.207 (70.2%). The pattern is already
suggestive — calibration helps least where the detector is weakest, because one cannot
calibrate away near-randomness. Table 1 reports pooled raw and calibrated ECE and Brier per
checkpoint. This foreshadows the central result: calibration's success is itself a function of
competence.

*[Table 1: per-checkpoint pooled ECE_raw, ECE_cal, Brier, with 95% bootstrap CIs.
Data: reports/calibration/xception_ffpp_pooled.csv, xception_FR_pooled.csv]*

---

## 5. The Competence–Calibration Coupling

### 5.1 Within FaceForensics++ (cross-forgery)

Holding architecture fixed (Xception) and varying both the training checkpoint (FS, FR) and the
FF++ manipulation method yields eight (competence, calibration) configurations. Calibration
error tracks competence closely: Pearson r = −0.82 (p = 0.013), Spearman ρ = −0.81 (p = 0.015),
with AUC spanning 0.39–0.97 and ECE_cal spanning 0.036–0.343. Where the detector is competent
(e.g. FS on FaceSwap, AUC 0.97) calibration is excellent (ECE 0.036); where it is incompetent
(e.g. FR on FaceSwap, AUC 0.39) calibration fails (ECE 0.33), even after the same calibration
procedure.

*[Table 2 rows for FF++; the full per-configuration table is Table 2.]*

### 5.2 Across the DF40 generator landscape

We next score the FS-Xception detector on eight DF40 generators spanning the FS, FR, and EFS
families, giving competence points from AUC 0.65 to 0.95. Considered alone, DF40 shows a
*weaker* association (Pearson r = −0.585, p = 0.13, not significant): calibration succeeds
across DF40's moderate-to-high competence range, with ECE_cal confined to 0.02–0.07 regardless
of AUC. Combining the DF40 and FF++ points (16 configurations) restores a strong, significant
coupling (r = −0.78, p = 4×10⁻⁴).

This juxtaposition raises an obvious and important question, visible in Figure 1: the DF40
points cluster at low calibration error across their AUC range, while the FF++ points span the
full range. Is the overall coupling driven by *competence*, or by a *dataset effect* — FF++
simply being harder to calibrate than DF40, and happening to span more competence? We address
this directly.

*[Figure 1: scatter of ECE_cal vs AUC, 16 points colored by dataset, OLS fit with bootstrap CI
band. File: figures/coupling_competence_calibration.png]*

### 5.3 Breaking the confound: competence manipulation

To separate competence from dataset, we *force* low competence on DF40 while holding the dataset
fixed. We score DF40's face-swap and synthesis generators (simswap, blendface, StyleGAN2, sd2.1)
with a deliberately *mismatched* reenactment-trained detector. This is a controlled
intervention: the data are unchanged, only the detector's competence is reduced by applying it
outside its training family. As intended, AUC collapses to 0.26–0.46.

The calibration result is unambiguous: all four forced-low-competence DF40 configurations show
*high* calibration error (ECE_cal 0.25–0.30), landing exactly among the FF++ low-competence
points (Figure 2). With these points included, the *within-DF40* coupling — absent when DF40
spanned only moderate-to-high competence — becomes strong and significant: Pearson r = −0.94
(p < 10⁻⁴) across twelve DF40 configurations. Because the dataset is held constant and only
competence is varied, the earlier flatness is explained: DF40 calibrated well not because it is
intrinsically easy, but because the matched detector kept its competence high. The dataset
confound is ruled out; the coupling is competence-driven.

*[Figure 2: scatter with the four FR-mismatch DF40 points (distinct color) filling the
low-AUC / high-ECE region. File: figures/coupling_with_FRmismatch.png]*

### 5.4 Architecture-generality and the full architecture x dataset grid

To test whether the coupling is specific to Xception, we repeat the analysis with
EfficientNet-B4, a different architecture, on *both* source datasets. On FF++ alone,
EfficientNet-B4 exhibits the coupling clearly (Pearson r = -0.95 over four configurations). We
then score the same EfficientNet-B4 detector on the eight DF40 generators, giving it the same
two-dataset coverage as Xception. Across both datasets (twelve configurations), EfficientNet-B4
shows a strong, significant coupling of its own (Pearson r = -0.77, p = 0.003), with its points
falling on the common trend rather than forming a separate one. Notably, on DF40 the
EfficientNet-B4 detector reaches lower competence than Xception did (AUC down to 0.47 on
StyleGAN2), supplying additional low-competence points within a single dataset and a single
architecture -- and these too exhibit the expected high calibration error.

Pooling all configurations -- two architectures, two source datasets, four forgery families,
and three checkpoint conditions, thirty-two points in total -- yields a strong, highly
significant coupling: Pearson r = -0.81 (p = 2.5x10^-8), Spearman rho = -0.82. Table 3
summarizes the correlation within each slice and combined; every slice that spans a non-trivial
competence range shows the negative coupling, and the only weak slice (DF40 with a matched
detector, where competence never falls below 0.65) is exactly the one whose competence range is
too narrow to exhibit it -- the case Section 5.3 resolves by extending that range within the
same dataset. Figure 3 shows the full grid: all thirty-two points, distinguished by
architecture and dataset, tracing a single descending relationship.

*[Figure 3 (headline): full architecture x dataset grid -- 32 points, markers by architecture
(Xception / EfficientNet-B4), colors by dataset (FF++ / DF40), OLS fit, r/p annotated.
File: figures/coupling_full_grid.png]*
*[Table 3: correlation per slice (FF++ Xception; DF40 FS-only; DF40 within-dataset; EfficientNet
FF++; EfficientNet both datasets; grand combined) with n, Pearson r/p, Spearman rho/p, AUC range.
Data: reports/calibration/TABLE3_coupling_summary.csv]*

### 5.5 Mechanism

The coupling admits a simple mechanistic account. A post-hoc calibrator learns the mapping from
raw score to empirical correctness *in the detector's competent regime* — the regime in which
the calibration data was generated. When that mapping is applied where the detector is
incompetent, it no longer holds: at low competence the score distribution and its relationship
to correctness have changed, so the learned correction is mismatched. Calibration can repair
miscalibration but cannot manufacture the competence on which a reliable score-to-correctness
mapping depends. Trust scores are therefore least reliable exactly where competence is lowest.

---

## 6. Discussion and Implications

**Deployment risk is concentrated where calibration fails.** Detector competence is lowest for
novel, unseen generators — precisely the inputs a deployed system most needs to handle and for
which honest uncertainty is most valuable. Our results show that a single static calibration is
least trustworthy on exactly these inputs, giving a false sense of reliability where the stakes
are highest. The failure is not random noise but a systematic, predictable degradation tied to
competence.

**Toward competence-aware trust scoring.** The natural response is to make the trust score's
reliability conditional on an estimate of competence for the input at hand — for example by
estimating competence per generation-family or per input region and conditioning the calibrated
score, or the abstention threshold, accordingly. This paper establishes the problem and its
lawful structure; a full competence-aware scoring system is the subject of ongoing work.

**Limitations.** Competence here is measured per generator and assumes generator labels at
evaluation time; deployment would require estimating competence without such labels. ECE is a
binned estimator with known sensitivity to bin count and scheme; we use equal-mass binning and
report robustness to the estimator in the appendix. Our evidence spans two architectures and two
source datasets — broad but not exhaustive — and uses image-level detectors rather than
video-level temporal models. None of these qualifications affects the central, multiply-replicated
finding.

---

## 7. Conclusion

The calibration of a deepfake trust score is coupled to the detector's competence —
robustly across architectures, datasets, and forgery families; within a single dataset; and
causally under competence manipulation. A deepfake detector's confidence cannot be trusted
uniformly: it is least trustworthy where the detector is weakest, which in deployment is where
it matters most. Trust scoring for deepfake detection must be made competence-aware.

---

## Figures and Tables

- **Figure 1** — ECE_cal vs AUC, 16 points by dataset (figures/coupling_competence_calibration.png)
- **Figure 2** — with FR-mismatch points breaking the confound (figures/coupling_with_FRmismatch.png)
- **Figure 3 (headline)** — full grid: 32 points by architecture x dataset (figures/coupling_full_grid.png)
- **Table 1** — pooled ECE_raw/ECE_cal/Brier per checkpoint
- **Table 2** — per-(detector, generator) AUC, ECE_raw, ECE_cal (all 24 base configurations)
- **Table 3** — correlation per slice + combined (reports/calibration/TABLE3_coupling_summary.csv)

## References (to complete)

Chollet 2017 (Xception); Tan & Le 2019 (EfficientNet); Rossler et al. 2019 (FF++); Li et al.
2020 (Celeb-DF); Yan et al. 2023 (DeepfakeBench); Yan et al. 2024 (DF40); Guo et al. 2017
(calibration / temperature scaling); Platt 1999 (Platt scaling); Zadrozny & Elkan 2002
(isotonic); Ovadia et al. 2019 (calibration under shift); Naeini et al. 2015 (ECE).
