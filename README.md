# Explainable AI for Cloud Incident Prediction using OpenTelemetry

**MSc Machine Learning and Artificial Intelligence**  
**Liverpool John Moores University — 2024–2026**

## Research Overview

This repository contains the implementation for my MSc dissertation 
investigating whether supervised machine learning can predict cloud 
incidents from operational log data before critical failures occur, 
and whether Explainable AI methods (SHAP and LIME) can produce 
faithful, consistent, and operationally actionable explanations 
for those predictions.

## Research Questions

- **RQ1:** How accurately can Logistic Regression, Random Forest, 
  and LightGBM predict cloud incidents from operational log data?
- **RQ2:** Do SHAP and LIME explanations remain faithful and 
  consistent, and do the two methods agree on influential features?
- **RQ3:** Do top-ranked features map to operationally recognisable 
  failure patterns for on-call engineers?

## Datasets

- **HDFS** — Hadoop Distributed File System logs (LogPai)
- **BGL** — Blue Gene/L supercomputer logs (LogPai)
- **OpenStack** — Cloud infrastructure logs (LogPai)

Source: https://github.com/logpai/loghub

## Technical Stack

| Component | Library | Version |
|-----------|---------|---------|
| Log parsing | drain3 | 0.9.11 |
| ML models | scikit-learn | 1.5.2 |
| Gradient boosting | lightgbm | 4.5.0 |
| SHAP explanations | shap | 0.46.0 |
| LIME explanations | lime | 0.2.0.1 |
| Data processing | pandas / numpy | 2.2.3 / 1.26.4 |

## Project Structure

```
xai-cloud-incident-prediction/
├── data/raw/          (datasets, not tracked in git)
├── data/processed/    (feature matrices, splits — not tracked in git)
├── src/               (all pipeline scripts, numbered in execution order)
├── results/           (CSV outputs; figures/, models/, shap_values/ subfolders)
├── logs/              (run logs, inventories, chapter outlines)
├── notebooks/         (exploratory work)
├── DISSERTATION_LOG.md
└── RESEARCH_DIARY.md
```

## Hardware and Environment

- **Hardware:** macOS Sequoia, Apple Silicon ARM64 (M-series chip)
- **Python:** 3.12.7, Conda environment `msc_dissertation`
- Run times below were measured on this hardware; a different machine
  (especially x86 without native ARM wheels for lightgbm/shap) may vary
  significantly.

## Scripts — Execution Order

Run in this numeric order; each script depends on outputs from earlier ones.

| # | Script | What it does | Approx. run time |
|---|--------|---------------|-------------------|
| 00 | `00_inspect_data.py` | Prints size, sample lines, and row/line counts for every raw and pre-processed dataset file | 1-2 min |
| 01 | `01_preprocess_bgl.py` | Runs Drain3 (with IP/hex/number masking) over BGL.log and builds the fixed-window (size=100) event-occurrence matrix | 5-8 min |
| 01 | `01_preprocess_openstack.py` | Runs Drain3 over all three OpenStack log files and builds the per-instance event-occurrence matrix | <1 min |
| 02 | `02_build_splits.py` | Builds chronological 70/15/15 train/val/test splits for HDFS and BGL; saves OpenStack's full matrix for k-fold instead | <1 min |
| 03 | `03_train_baselines.py` | Trains Logistic Regression, Random Forest, LightGBM on all three datasets and saves RQ1 metrics | 2-5 min |
| 04 | `04_explain_rq2.py` | Trains the best model per dataset (HDFS: LightGBM, BGL: Random Forest), runs SHAP + LIME, faithfulness tests, and SHAP-LIME Spearman agreement | 10-15 min |
| 05 | `05_explain_openstack_shap.py` | Trains a regularised LightGBM on the full OpenStack matrix and runs SHAP for qualitative RQ3 analysis only | <1 min |
| 06 | `06_rq3_actionability.py` | Maps HDFS/BGL top-15 SHAP features to operational categories via keyword rules; computes actionability coverage | <1 min |
| 07 | `07_rq3_openstack_qualitative.py` | Derives and categorises the event templates triggered by OpenStack's 4 anomalous instances | <1 min |
| 08 | `08_ablation_window_size.py` | Rebuilds BGL windows at sizes 10/20/40 (window_size=100 reuses the existing split) and evaluates RF/LightGBM at each | 5-10 min |
| 09 | `09_shap_stability_window.py` | Retrains RF at each BGL window size, runs SHAP, and computes pairwise Jaccard similarity of top-10 features across window sizes | 10-15 min |
| 10 | `10_temporal_drift_bgl.py` | Splits BGL's chronological test set into 3 sequential chunks and evaluates RF/LightGBM on each | 2-3 min |
| 11 | `11_hyperparameter_sensitivity.py` | 27-combination LightGBM grid search (n_estimators x learning_rate x num_leaves) on BGL | 5-10 min |
| 12 | `12_results_inventory.py` | Lists every file in results/ with size/schema/preview and checks 15 expected deliverables are present | <1 min |
| 13 | `13_figure_inventory_check.py` | Verifies required dissertation figures exist at dpi>=150 with titles; regenerates any missing ones | 5-8 min |

Shared helper modules (not run directly): `utils_data.py`, `utils_explain.py`, `utils_bgl_windows.py`, `utils_rq3.py`.

## Results CSVs

| File | Contents |
|------|----------|
| `results/model_results.csv` | RQ1: F1/Precision/Recall/AUC-ROC per dataset/model/eval-split (val, test, or cv_mean/cv_std for OpenStack) |
| `results/feature_ranking_{HDFS,BGL,OpenStack}.csv` | Per-feature mean absolute SHAP (and LIME, where applicable) importance, sorted descending |
| `results/lime_ranking_{HDFS,BGL}.csv` | Per-instance LIME feature weight matrix for the 200-sample test set |
| `results/faithfulness_{HDFS,BGL}.csv` | Deletion-test mean/std probability drop at k=1,2,3,5,10 for SHAP, LIME, and a random-ranking control |
| `results/shap_lime_spearman.csv` | RQ2: Spearman rank correlation between SHAP and LIME global feature importances, per dataset |
| `results/rq3_actionability_{HDFS,BGL}.csv` | Top-15 SHAP features with template text and assigned operational category |
| `results/rq3_coverage.csv` | RQ3 actionability_coverage summary (categorised / 15) per dataset |
| `results/rq3_openstack_qualitative.txt` | Qualitative-only RQ3 writeup for OpenStack (no held-out metrics) |
| `results/ablation_window_size.csv` | RQ1 follow-up: F1/Precision/Recall/AUC-ROC for RF/LightGBM at BGL window sizes 10/20/40/100 |
| `results/temporal_drift_bgl.csv` | F1/Precision/Recall/AUC-ROC for RF/LightGBM on each of 3 sequential chunks of BGL's test set |
| `results/hyperparameter_sensitivity.csv` | Test F1/AUC-ROC for all 27 LightGBM hyperparameter combinations on BGL |
| `results/shap_stability_window.csv` | Pairwise Jaccard similarity of top-10 SHAP features across BGL window sizes |