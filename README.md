# CPS Anomaly Detection Framework

**Research project:** A Simulation-Based Framework to Enhance the Accuracy of Anomaly Detection Models in Cyber-Physical Systems

**Institution:** Dalian University of Technology — School of Software Technology
**Researcher:** BERETE MAMADY
**Supervisor:** Prof. Li Mingchu

This repository documents a controlled experimental study of anomaly detection and model-specific retraining in cyber-physical systems (CPS). The smart-power-grid setting is an application-oriented interpretation of the evaluated data perturbations, **not** evidence of validation on a dedicated power-flow simulator.

> **Reproducibility status:** The preserved result tables support auditing and figure regeneration. Full model retraining requires the exact external dataset and historical intermediate result files. The repository is not claimed to offer a single end-to-end command that reproduces every published score from scratch.

## Experimental protocol

- Final combined dataset: **1,441,719 rows**, **51 numeric model-input features**, with **1,387,098 normal** and **54,621 anomalous** rows.
- For each split seed (`42`, `123`, `2024`, `2025`, `3407`): stratified **60% train / 20% retrain / 20% held-out test**.
- Split sizes: **865,031 / 288,344 / 288,344**. Classical models use 288,344 test rows; LSTM forms length-10 sequences within the test partition and evaluates **288,335 endpoints**.
- Benchmark models: XGBoost, Random Forest, Linear SVM, LSTM, Autoencoder, Isolation Forest and One-Class SVM.
- Detailed final ablation: four models (XGBoost, Random Forest, Linear SVM and LSTM), five split seeds, and three conditions: baseline, retraining without additional weighting, and weighted retraining.
- Final anomaly weights: **3.0** for XGBoost, Random Forest and Linear SVM; **1.0** for LSTM. Weight 3.0 is the retained configuration for the classical models, **not** a claimed global optimum established by a test-set sweep. LSTM 1.0 was selected as a candidate using internal-validation, five-seed weight sensitivity.

## Final five-seed summary

| Model | Baseline mean F1 | Final mean F1 | Change | Mean FN reduction |
|---|---:|---:|---:|---:|
| XGBoost | 0.95082 | 0.96270 | +0.01188 | 361.8 |
| Random Forest | 0.90846 | 0.94086 | +0.03240 | 621.0 |
| Linear SVM | 0.83344 | 0.85753 | +0.02409 | 524.4 |
| LSTM | 0.96867 | 0.97259 | +0.00391 | 46.4 |

These values must be traced to the final four-model ablation outputs, not to earlier explorations. In the LSTM ablation, the `weighted_retraining` row is **derived from the identical `retraining_without_weighting` configuration** when anomaly weight = 1.0; it is not a separate LSTM training run.

## Results and provenance

Key result artifacts in the original working tree include:

- `results/adaptive/final_validation/ablation/metrics/ablation_results.csv` — 30 historical XGBoost / Random Forest rows (2 models × 5 seeds × 3 conditions).
- `results/adaptive/final_validation_four_model/ablation/metrics/four_model_ablation_results.csv` — consolidated final 60 rows: reused XGBoost / Random Forest results, computed Linear SVM and LSTM runs, and derived LSTM weight-1 rows.
- `results/adaptive/lstm_weight_sensitivity_multiseed/metrics/lstm_weight_multiseed_results.csv` — 25 internal-validation outcomes across five weights and five split seeds; selected candidate weight = 1.0.
- `results/adaptive/final_validation_four_model/report_extensions/weight_sensitivity/four_model_weight_sensitivity_results.csv` — **retrospective seed-42 diagnostic**, not the source of the main five-seed performance claim or an unbiased weight-selection procedure.
- `results/adaptive/final_validation_four_model/figures/final_four_model_figure_manifest.csv` — figure-to-source manifest. Figure numbers in the manuscript are not the same as numbers in this experimental manifest.

Script roles in the original tree:

- `src/adaptive_feedback/final_validation/ablation_analyzer.py`: original two-model ablation results.
- `src/adaptive_feedback/final_validation/four_model_ablation_extension.py`: extends and consolidates the four-model ablation; **depends on pre-existing result files**.
- `src/adaptive_feedback/final_validation/lstm_weight_sensitivity_multiseed.py`: LSTM internal-validation candidate weight selection.
- `src/adaptive_feedback/final_validation/four_model_report_extensions.py`: retrospective diagnostics and transitions.
- `src/adaptive_feedback/final_validation/generate_final_four_model_report_figures.py`: generates figures from preserved result files; can **overwrite outputs**, so first work in a disposable copy.
- `src/adaptive_feedback/final_validation/multi_seed_validator_four_model.py`: historical validation containing an earlier LSTM weighting configuration; **do not use its weighted LSTM values as final values**.

The scripts and artifacts listed above are included in the planned publication selection. Availability in a public checkout depends on completion of the first Git commit and push.

## Dataset identity and access

Original expected relative path:

`results/adaptive/retraining/datasets/retraining_dataset_combined.csv`

Dataset SHA-256 (calculated locally with Windows `Get-FileHash`):

`a3f18ef39df7815ea3fa7071259b35a919a940cc5617df3669f5551258abc97a`

The large dataset is **not** tracked in ordinary Git. A public download link, license/redistribution permissions, original source, and/or exact reconstruction procedure **still need to be documented**. Without accessible identical data, independent full retraining is not currently available. The short 32-character fingerprint stored by some scripts is **not** the complete SHA-256 string above.

To validate a separately obtained dataset on Windows PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 -Path ".\results\adaptive\retraining\datasets\retraining_dataset_combined.csv"
```

## Python environment

The supplied environment export reflects the **current** `Jupyter_3.11` environment, not independently verified package versions from each earlier experiment. Relevant observed versions include Python 3.11.14, NumPy 2.4.3, pandas 3.0.1, scikit-learn 1.8.0, XGBoost 3.2.0, TensorFlow 2.21.0, Keras 3.13.2, Matplotlib 3.10.8, SciPy 1.17.1 and joblib 1.5.3.

Use a reviewed project environment file; do not publish a machine-specific Conda `prefix:` line. Installation and reproduction commands must be tested in a **separate fresh environment** before claiming they are verified.

## Important limitations

- Current results support controlled **stratified split** stability, not independent temporal-deployment or external-domain generalization.
- The LSTM's local sequences formed within stratified partitions do not necessarily represent real chronological continuity.
- Smart-grid mapping is interpretive; no grid physics or controller safety is measured.
- The current final F1 gains cannot be causally attributed to newly generated difficulty-aware hard samples; measured gains reflect controlled retraining and model-specific weighting.
- The retrospective test-set sensitivity and threshold sweeps are **diagnostic**. They must not be presented as unbiased parameter selection or as a substitute for the fixed five-seed results.
- Exact paired final LSTM prediction transitions were not preserved and are **not available**.

## Reproducibility and availability

The selected scripts, tables and figures allow independent inspection of the reported metrics and supporting evidence. The large combined dataset is excluded from version control. Full reproduction from raw data remains dependent on access to the exact dataset and the historical intermediate outputs described above. Installation in a fresh environment and end-to-end reruns have not been independently verified. No new experimental results are claimed by this repository.
