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