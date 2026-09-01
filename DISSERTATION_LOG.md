# Dissertation Research Log

**Candidate:** Mintu Ghosh
**Title:** Explainable AI for Cloud Incident Prediction using OpenTelemetry
**Programme:** MSc Machine Learning and Artificial Intelligence — LJMU
**Start date:** 05 August 2026

---

## Environment

| Item | Value |
|------|-------|
| OS | macOS Sequoia — Darwin ARM64 (Apple Silicon) |
| Python | 3.12.7 |
| Conda env | msc_dissertation |
| Key libraries | scikit-learn 1.5.2, lightgbm 4.5.0, shap 0.46.0, lime 0.2.0.1, drain3 0.9.11 |
| requirements.txt committed | Yes |

---

## Dataset Log

| Dataset | Source | Downloaded | File size | Anomaly rate |
|---------|--------|------------|-----------|--------------|
| HDFS | logpai/loghub | 05 Aug 2026 | 1.58 GB | ~2.9% |
| BGL | logpai/loghub | 05 Aug 2026 | 743 MB | ~8% |
| OpenStack | logpai/loghub | 05 Aug 2026 | 61 MB | ~0.2% (4 sessions) |

---

## Error Log

| Date | Script | Error | Root cause | Fix applied |
|------|--------|-------|------------|-------------|
| 05 Aug 2026 | git push | Push rejected | GitHub initialised with README, local was behind | git pull origin main --allow-unrelated-histories |
| 05 Aug 2026 | terminal | zsh: command not found: import | Ran Python code directly in shell | Use python script.py instead |
| 05 Aug 2026 | lime verify | AttributeError: module lime has no __version__ | lime does not expose __version__ | Verified with pip show lime — version 0.2.0.1 confirmed |
| 07 Aug 2026 | 04_explain_rq2.py | SHAP additivity check failure on BGL RF | class_weight=balanced breaks TreeExplainer internal assumptions | Disabled strict additivity check — documented workaround |
| 07 Aug 2026 | 05_explain_openstack_shap.py | E3 SHAP magnitude pathological (10,632) | LightGBM overfitting to 4 anomaly points with scale_pos_weight=516 | Regularised model with shallower trees and sqrt class weight |

---

## Decision Log

| Date | Decision | Reason | Reference |
|------|----------|--------|-----------|
| 05 Aug 2026 | Use pre-processed HDFS Event_occurrence_matrix.csv | LogPai already ran Drain3 on HDFS — results directly comparable to published benchmarks | He et al. (2016) |
| 05 Aug 2026 | Run Drain3 myself on BGL and OpenStack | No pre-processed files available for these datasets | He et al. (2016) |
| 05 Aug 2026 | Stratified 5-fold CV for OpenStack only | Normal (2017-05-16/17) and abnormal (2017-05-14) logs are from disjoint time periods — chronological split puts 100% of anomalies in train | Documented methodological exception in Chapter 3 |
| 05 Aug 2026 | Chronological 70/15/15 split for HDFS and BGL | Both datasets have temporally ordered sessions — prevents data leakage | Aligned with proposal Stage 2 |
| 06 Aug 2026 | Best model HDFS — LightGBM | Highest F1 (0.999) and AUC-ROC (0.999) on test set | RQ1 results |
| 06 Aug 2026 | Best model BGL — Random Forest | Highest test Precision (0.97) and best overall test F1 (0.47). High-precision alerts are more operationally trustworthy in SRE context | RQ1 results |
| 06 Aug 2026 | OpenStack excluded from RQ1/RQ2 quantitative evaluation | Only 4 anomalous sessions — insufficient for held-out evaluation | Documented in Chapter 3 Limitations |
| 06 Aug 2026 | Dropped Type column from HDFS before training | Prevents direct label leakage into features | Data integrity |

---

## Daily Log

---

### Day 1 — 05 August 2026

**Objective:** Environment setup, datasets downloaded, pipeline started

**Completed:**
- [x] conda env msc_dissertation created
- [x] All libraries installed and verified on ARM64
- [x] Project folder structure created
- [x] GitHub repo created: xai-cloud-incident-prediction
- [x] HDFS dataset downloaded (1.58 GB)
- [x] BGL dataset downloaded (743 MB)
- [x] OpenStack dataset downloaded (61 MB)
- [x] Claude Code running with master prompt
- [x] HDFS chronological split built
- [x] BGL chronological split built
- [x] OpenStack — stratified 5-fold CV decided

**Key finding today:**
OpenStack has only 4 anomalous sessions out of 2,069 total. The
abnormal and normal logs were collected on different dates with no
time overlap. Chronological split is structurally impossible for
this dataset. This is a known limitation of the OpenStack LogPai
dataset, not a code error. Documented as methodological exception.

**My observation:**
The realisation that OpenStack log data had no chronological
overlap between normal and abnormal states was a sharp reminder
that raw academic datasets rarely match real-world continuous
deployment scenarios. It forced me to pivot my methodology on day
one, and catching the Drain3 template explosion (3,983 templates
before masking UUIDs and IPs) saved me from feeding garbage
high-dimensional noise into the models later. This is a genuine
methodological finding.

---

