# deepfake-trust-research

## Correction (Aug 2026)

The preprint v1's central negative claim was an artifact of a tie-handling degeneracy in the stable-sort equal-mass ECE estimator (nearest prior observation: Pernot, arXiv:2306.05180, for regression-uncertainty binning; the label-ordered classification case here is its deterministic, selective extreme). `src/metrics.py` is patched (tie-safe default; legacy retained as `equal_mass_legacy`); a constant-prevalence regression test is included. Corrected results and full audit: see manuscript v2 (arXiv:2606.29484v2). Audit scripts and their outputs live in `audits/`.

Calibrated Deepfake Trust Score (CDTS) - a self-auditing trust instrument for
deepfake detection under Graceful Trust Degradation (GTD).

**Split:** code/notebooks/results/figures -> GitHub ; frames/logits/weights -> Drive (gitignored).
**Paths:** read from `config/paths.yaml` only.
**Reproducibility:** seeds + bootstrap_B + ECE binning + alarm rule locked in `config/experiment.yaml`;
DeepfakeBench pinned at commit `f188b1c105465e2e5377eb536a95022ae0e4522d`.

**Layout:** `notebooks/` (numbered analysis stages), `src/` (shared modules), `audits/`
(estimator-audit scripts and outputs), `reports/` (result CSVs), `figures/` (manuscript
figures, `v3_fig_*` current), `legacy/` (pre-correction drafts, figures, and derived
tables, preserved for the record; see `legacy/README.md`).
