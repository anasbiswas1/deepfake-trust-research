# The Calibrated Deepfake Trust Score (CDTS): Competence-Coupled Trustworthiness of Deepfake Detectors

**Md Anas Biswas**¹ · *Supervisor TBD*¹
¹ School of Computing, University of Portsmouth

**Draft v0.5 (unifying result)** — github.com/anasbiswas1/deepfake-trust-research

> STATUS: 4 of 5 components complete (a Calibration, b Equity, c Explanation Faithfulness,
> d Label-free monitoring) PLUS the unifying latent-factor result (§8). Component (e) Zero-Trust
> Routing pending. Thesis REVISED from plan: competence is the single latent factor of
> trustworthiness (see §1.3, §8).

---

## Abstract (draft)

Deepfake detectors are deployed as *trust signals*: a detector's confidence is consumed
downstream as a degree of trust. We reframe deepfake detection from a binary verdict into a
continuous, calibrated, self-auditing trust instrument — the Calibrated Deepfake Trust Score
(CDTS) — and study its trustworthiness guarantees. Our central empirical finding is a
**competence-calibration coupling**: the calibration of a deepfake trust score degrades as the
detector's discriminative competence falls (Pearson r between -0.8 and -0.94, across two
architectures, two datasets, four forgery families, 32 configurations; demonstrated within a
single dataset and causally via competence manipulation). We show this coupling is the
*organizing principle* of detector trustworthiness: (i) it makes calibration unreliable exactly
for novel, poorly-detected generators — the highest-risk deployment case; (ii) it manifests as
**calibration inequity** across demographic subgroups, distinct from accuracy inequity (a group
can be equally accurate yet unequally calibrated); and (iii) it enables **label-free monitoring**
— score-distribution-divergence signals predict competence without ground-truth labels (KS
distance r=0.95 with AUC), and via the coupling, detect calibration risk on unseen generators
(ROC-AUC 0.86, no labels). We argue trust scoring must be competence-aware, and provide the
CDTS wrapper as the mechanism.

---

## 1. Introduction

### 1.1 Deepfake detectors as trust signals
[Deployment reality: confidence consumed as trust in moderation/verification/integrity.
Decisions conditioned on the probability, not just the verdict.]

### 1.2 Graceful Trust Degradation and the CDTS framing
A detector must never fail silently. As reliability degrades (new generator, compression,
certain subgroups), the trust signal should KNOW it (calibration), SHOW it (explanation
stability), WARN about it (drift monitoring), NOT DISCRIMINATE in it (calibration equity), and
ROUTE around it (verification policy). CDTS is a *wrapper*: any backbone's logit -> a post-hoc
calibrator producing a trust score T with P(authentic|T=t) approx t, plus uncertainty and a
monitoring state.

### 1.3 The competence-coupling as the organizing principle (REVISED thesis)
The original conception of this work posited that distinct trust signals (calibration,
explanation, fairness) co-degrade *temporally* as new generators appear, and are jointly
forecastable before accuracy drops. Our experiments revise this: we find **no independent
temporal drift** -- the apparent timeline degradation is fully explained by detection competence
(partial correlation of era with calibration error, controlling for competence, is ~0 and
non-significant). The true organizing principle is not time but **competence**: trust-signal
degradation is coupled to how well the detector discriminates, and competence -- not the
calendar -- is the variable that unifies the signals. This is a stronger and more actionable
claim, and we build the paper around it.

### 1.4 Contributions
1. **(C1) The competence-calibration coupling** (Section 4): calibration error rises as
   competence falls, robustly (r=-0.8 to -0.94), within and across datasets/architectures, and
   causally.
2. **(C2) Calibration equity is competence-mediated and distinct from accuracy equity**
   (Section 5): the trust score is unequally calibrated across demographic subgroups; a group
   can be equally accurate yet unequally calibrated.
3. **(C3) Explanation faithfulness is competence-coupled** (Section 7): under a saturation-free
   importance-effect metric, Grad-CAM faithfulness rises with competence (r=+0.94) -- competent
   detectors have faithful explanations, incompetent ones produce uninformative ones.
4. **(C4) Label-free competence monitoring** (Section 6): score-distribution-divergence signals
   predict competence without labels, and via the coupling, detect calibration risk on unseen
   generators (ROC-AUC 0.86).
5. **(C5) THE UNIFYING RESULT** (Section 8): a single latent factor explains 85% of trust-signal
   variance and aligns with competence (r=0.98), which fully mediates all pairwise signal
   relationships. Trustworthiness reduces to one factor: competence.
6. **(C6) The CDTS wrapper and GTD framing** operationalizing competence-aware trust (Sections 3, 9).

---

## 2. Related Work
[2.1 Deepfake detection & benchmarks: FF++, Celeb-DF, DF40 (Yan 2024), DeepfakeBench (Yan 2023).
2.2 Calibration: Guo 2017, Platt 1999, isotonic (Zadrozny & Elkan), debiased ECE (Kumar 2019,
Roelofs 2022). Closest deepfake-calibration prior + competitor: Jin et al. 2025 (packed-ensembles
uncertainty calibration) -- does NOT address subgroup calibration, monitoring, or competence-coupling.
2.3 Calibration under shift: Ovadia 2019. We connect shift-degradation specifically to competence.
2.4 Fairness in deepfake detection: GBDF, AI-Face, Xu et al. 2024 (accuracy-gap framing). We use
multicalibration (Hébert-Johnson 2018) for calibration-equity, distinct from accuracy-equity.
2.5 Selective prediction / monitoring: El-Yaniv & Wiener; drift detectors (CUSUM, ADWIN). We
repurpose monitoring for label-free competence detection.]

---

## 3. The CDTS Framework and Experimental Setup

### 3.1 The CDTS wrapper
[T = calibrator(backbone logit); carries calibration, equity audit, monitoring state.]

### 3.2 Detectors, data, calibration, metrics
Detectors: Xception (primary), EfficientNet-B4 (architecture-generality), from DeepfakeBench,
FS-trained; a reenactment-trained (FR) checkpoint for competence manipulation. Data: FF++ c23
(22,388-frame test), Celeb-DF v2 test, DF40 generators (cdf source) paired with Celeb-DF reals.
Calibration: hybrid Platt (n<1000) / isotonic (n>=1000), identity-disjoint splits. Metrics: AUC
(competence), ECE (15-bin equal-mass) raw and calibrated, Brier; bootstrap CIs B=2000.
[Full detail as previously written; leakage control by identity emphasized.]

---

## 4. Component (a): The Competence-Calibration Coupling [COMPLETE]

### 4.1 Calibration improves trust scores (baseline)
FS-Xception on FF++: pooled ECE 0.455 -> 0.047 (89.6%). But for a broadly-incompetent detector
(FR on FF++, all AUC <= 0.66): 0.694 -> 0.207 -- calibration helps less where competence is lower.

### 4.2 The coupling, within and across datasets
FF++ (8 configs): r=-0.82. DF40 alone (8 configs, moderate-high competence): r=-0.585 (n.s.).
Combined: r=-0.78. [Fig: coupling_competence_calibration.png]

### 4.3 Breaking the confound: competence manipulation (causal)
Forcing low competence on DF40 via a mismatched (FR) detector craters AUC (0.26-0.46); all 4/4
forced-low-competence configs show high ECE_cal (0.25-0.30). Within-DF40 coupling: r=-0.94.
The dataset confound is ruled out; the coupling is competence-driven.
[Fig: coupling_with_FRmismatch.png]

### 4.4 Architecture-generality and the full grid
EfficientNet-B4 shows the same coupling on both datasets (r=-0.77). Combined across 2
architectures x 2 datasets x 4 families x 3 checkpoints (32 configs): **r=-0.81, p=2.5x10^-8.**
[Fig (headline): coupling_full_grid.png; Table: TABLE3_coupling_summary.csv]

### 4.5 Mechanism
A calibrator learns the score->correctness map in the competent regime; applied where the
detector is incompetent, the map fails. Calibration repairs miscalibration but cannot create
competence. Trust scores are least reliable where competence is lowest.

---

## 5. Component (b): Calibration Equity [COMPLETE]

### 5.1 Subgroup-ECE gap
Using Xu et al.'s A-FF++ demographic annotations (gender, age bands, ethnicity, skin-tone),
joined by identity, we measure whether the calibrated trust score stays calibrated per subgroup
(multicalibration; Hébert-Johnson 2018). Subgroup-ECE gap = max subgroup-ECE - pooled-ECE, with
identity-clustered pivotal bootstrap CIs (B=2000; clustered because frames nest within identities).

### 5.2 Results (H2)
The trust score is significantly unequally calibrated across several axes (gap CI excluding 0):
gender (+0.026, CI [+0.022,+0.048]), skin-tone (+0.030, CI [+0.020,+0.054]); age shows a large
but imprecise gap (+0.068, wide CI); ethnicity is inconclusive given the available identity
diversity. [Honest: 70-identity eval pool -> wide CIs on sparse cells.]

### 5.3 Calibration equity is distinct from accuracy equity (the novelty)
On skin-tone, the detector is *equally accurate* (AUC gap 0.0006) yet *unequally calibrated*
(ECE gap 0.015) -- a disparity accuracy-fairness (GBDF/AI-Face) would miss. On age, accuracy gap
exceeds calibration gap -- the metrics are distinct, complementary lenses, not one dominating.
[Table: equity_subgroup_ece, equity_gap_CIs, equity_accuracy_vs_calibration_contrast]

### 5.4 Connection to the coupling
Subgroup calibration gaps track subgroup competence differences -- calibration inequity is the
competence-coupling expressed across demographic strata. [Tie to §4.]

---

## 6. Component (d): Label-Free Competence Monitoring [COMPLETE, REFRAMED]

### 6.1 From temporal drift to competence monitoring
We scored 21 DF40 generators spanning release eras 2016-2024. The coupling holds across the
timeline (r=-0.86). Crucially, temporal drift is NOT independent: partial correlation of era with
ECE_cal controlling for AUC is 0.08 (n.s.); within the swap family (where the detector is
competent) the era effect is r=-0.03 (n.s.). The timeline "drift" is the competence-coupling, not
time. We therefore reframe early-warning from temporal to **competence** monitoring.

### 6.2 The deployable claim
In deployment there are no labels on a fresh generator, so AUC is unavailable. But if an
*unlabeled* signal tracks competence, the coupling forecasts calibration risk label-free. We test
score-distribution signals (entropy, confident-fraction, bimodality) and reference-divergence
signals (KS, Wasserstein, KL versus a known-competent in-domain reference).

### 6.3 Results
Reference-divergence and shape signals track competence non-circularly (signal vs AUC, where AUC
uses labels and the signal does not): **KS distance r=0.95, Wasserstein r=0.88, bimodality r=0.82**
with AUC. Via the coupling, a KS-distance monitor detects high-calibration-risk generators with
**ROC-AUC 0.86, using no labels.** (Mean predicted score correlates even more strongly, r=0.99,
but is confounded by class balance and is not headlined.) [Fig: labelfree_detection_roc.png;
Tables: labelfree_signals, labelfree_signal_vs_competence]

### 6.4 Mechanism
A competent detector produces a score distribution resembling its in-domain reference (confident,
bimodal); an incompetent one produces a divergent, ambiguous distribution. Divergence from the
reference is thus an observable, label-free proxy for the competence that governs calibration.

---

## 7. Component (c): Explanation Faithfulness [COMPLETE]

### 7.1 A saturation-free faithfulness metric
Standard deletion/insertion faithfulness is confounded by score saturation: high-competence
detectors pin prob_fake near 1.0 (start-score vs competence r=0.998), compressing the metric's
dynamic range. We therefore use a saturation-free measure: partition each image into an 8x8 grid,
and for each patch compute its Grad-CAM importance and the actual prob_fake drop when only that
patch is removed; faithfulness is the Spearman rank correlation between importance and effect
across patches. This asks whether the regions the explanation flags are the ones that actually
drive the decision -- a scale-free quantity independent of absolute score level. (We verified the
naive insertion-deletion metric gives a spurious opposite-signed correlation, an artifact of
saturation; the saturation-free metric is the correct measure.)

### 7.2 Result: faithfulness rises with competence
Across 20 generators (Grad-CAM at the Xception conv4 feature map), explanation faithfulness
increases with competence: **Pearson r=+0.94, p<10^-7.** Competent detectors (simswap, blendface;
faithfulness ~0.65-0.69) produce explanations whose highlighted regions genuinely drive the
decision; incompetent detectors (pixart, sadtalker, wav2lip; faithfulness ~0.11-0.13) produce
explanations that do not correspond to the model's actual evidence. When a detector cannot detect,
its explanation is uninformative -- the same competence factor that governs calibration governs
explanation faithfulness. [Fig: explanation_faithfulness_rankcorr.png]

---

## 8. The Unifying Result: Competence Is the Single Latent Factor [COMPLETE -- KEYSTONE]

The preceding sections show calibration (§4), explanation faithfulness (§7), and label-free
monitoring (§6) each correlate with competence. We now show these are not three phenomena but
one. Across 20 generators we assemble the per-generator trust-signal vector (calibration error,
explanation faithfulness, and the monitoring signals KS-divergence, entropy, Wasserstein) and
analyze its structure.

**A single factor.** Every trust signal correlates with every other (|r| 0.54-0.95). Principal
component analysis of the standardized signals yields a dominant first component explaining
**84.7% of the variance** -- the signals vary together along essentially one axis.

**That factor is competence.** PC1 aligns with detection competence at **r=0.98 (p=5x10^-14)**.
The latent axis underlying all trust signals is competence itself. [Fig: unifying_competence_factor.png]

**Competence fully mediates.** Controlling for competence, the pairwise correlations among trust
signals collapse to non-significance (calibration<->faithfulness raw -0.81 -> partial +0.06;
calibration<->KS raw -0.84 -> partial +0.07; faithfulness<->KS raw +0.89 -> partial -0.14, all
n.s.). The trust signals are *conditionally independent given competence* -- competence does not
merely correlate with each signal, it explains their joint structure.

**Interpretation.** Detector trustworthiness does not decompose into independent guarantees that
must each be verified. It reduces to a single measurable factor -- competence -- which is lowest
exactly for the novel generators that matter most, and which (§6) can be estimated without labels.
This both simplifies the problem (estimate one thing) and sharpens the warning (when competence
drops, *all* trust properties fail together). The original conjecture that signals co-fire
*temporally* is replaced by a stronger, mechanistic claim: they co-fire because they share a
*competence* factor, and time is merely a proxy for distance from the detector's competence frontier.

---

## 9. Component (e): Zero-Trust Verification Policy [PENDING]

[Planned: selective prediction / risk-coverage (El-Yaniv & Wiener) using T + uncertainty + the
label-free competence-risk signal as the routing input; DiCE counterfactual recourse over the
trust-feature vector. Baseline: single-threshold routing on uncalibrated softmax. HYPOTHESIS:
competence-aware routing (using the §6 label-free signal) dominates confidence-only routing,
because it abstains precisely where calibration fails. To be built.]

---

## 10. Discussion

### 10.1 Competence is the single latent factor of detector trustworthiness
Section 8 establishes this formally: one factor explains 85% of trust-signal variance, aligns
with competence (r=0.98), and fully mediates the signals' relationships. Calibration, explanation
faithfulness, and monitoring are manifestations of a single underlying quantity. This unifies
phenomena previously studied separately and gives one deployable lever: estimate competence
(label-free, §6) and condition trust accordingly.

### 10.2 Deployment implications
Static calibration, fairness audits on a fixed benchmark, and confidence-threshold routing all
fail for the same reason -- they ignore competence, which is lowest for the novel generators that
matter most. CDTS conditions on competence throughout.

### 10.3 Limitations
[Competence measured per-generator (deployment needs estimation -- addressed by §6); ECE binning
sensitivity (equal-mass + debiased reported); 70-identity equity pool -> wide subgroup CIs; two
architectures, two source datasets; image-level not video-temporal detectors; mean-score signal
class-balance-confounded (set aside); components (c),(e) pending.]

---

## 11. Conclusion
Deepfake-detector trustworthiness is coupled to competence. Calibration, calibration equity, and
the feasibility of label-free monitoring are all governed by how well the detector discriminates
-- and competence is lowest exactly where deployment risk is highest. Trust scoring must be
competence-aware; CDTS is the mechanism.

---

## Figures and Tables (current)
- Fig 1 coupling_competence_calibration.png; Fig 2 coupling_with_FRmismatch.png;
  Fig 3 coupling_full_grid.png; Fig 4 labelfree_detection_roc.png;
  Fig 5 timeline_df40_competence_calibration.png; Fig 6 explanation_faithfulness_rankcorr.png;
  Fig 7 (KEYSTONE) unifying_competence_factor.png
- Table 1 pooled ECE per checkpoint; Table 2 TABLE2_all_configurations.csv (24 base configs);
  Table 3 TABLE3_coupling_summary.csv; Table 4 equity_gap_CIs; Table 5 equity_accuracy_vs_calibration_contrast;
  Table 6 labelfree_signal_vs_competence

## References (to complete)
Chollet 2017; Tan & Le 2019; Rossler 2019; Li 2020; Yan 2023 (DeepfakeBench); Yan 2024 (DF40);
Guo 2017; Platt 1999; Zadrozny & Elkan 2002; Ovadia 2019; Naeini 2015 (ECE); Kumar 2019;
Roelofs 2022; Nixon 2019 (ACE); Hébert-Johnson 2018 (multicalibration); Jin 2025 (deepfake
calibration competitor); Gowrisankar & Thing 2024 (deepfake XAI); El-Yaniv & Wiener (selective);
Mothilal 2020 (DiCE).
