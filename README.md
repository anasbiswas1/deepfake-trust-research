# deepfake-trust-research
Calibrated Deepfake Trust Score (CDTS) - a self-auditing trust instrument for
deepfake detection under Graceful Trust Degradation (GTD).

**Split:** code/notebooks/results/figures -> GitHub ; frames/logits/weights -> Drive (gitignored).
**Paths:** read from `config/paths.yaml` only.
**Reproducibility:** seeds + bootstrap_B + ECE binning + alarm rule locked in `config/experiment.yaml`;
DeepfakeBench pinned at commit `f188b1c105465e2e5377eb536a95022ae0e4522d`.

Notebooks: 00_setup, 01_data_preprocess, 02_backbones, 03_calibration,
04_calibration_equity, 05_explanation_stability, 06_drift_warning,
07_routing, 08_coupling, 09_paper_tables, 10_baselines, 11_deliverable.
