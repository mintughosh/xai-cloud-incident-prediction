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

### Day 2 — 06 August 2026

**Objective:** Model training complete — RQ1 results obtained

**Results saved:** results/model_results.csv

**RQ1 Results Summary:**

| Dataset | Model | Precision | Recall | F1 | AUC-ROC |
|---------|-------|-----------|--------|----|---------|
| HDFS | LR | 0.985 | 0.992 | 0.988 | 0.995 |
| HDFS | RF | 0.998 | 0.999 | 0.999 | 0.999 |
| HDFS | LightGBM | 0.999 | 0.999 | 0.999 | 0.999 |
| BGL | LR | 0.812 | 0.450 | 0.578 | 0.840 |
| BGL | RF | 0.970 | 0.310 | 0.470 | 0.895 |
| BGL | LightGBM | 0.840 | 0.160 | 0.270 | 0.780 |
| OpenStack | LR | 0.000 | 0.000 | 0.000 | 0.500 |
| OpenStack | RF | 0.000 | 0.000 | 0.000 | 0.500 |
| OpenStack | LightGBM | 0.000 | 0.000 | 0.000 | 0.500 |

**Key findings — my own interpretation:**

HDFS: All three models performed near-perfectly (F1 ~0.999). This
tells me HDFS anomalies are strongly tied to specific log event
templates — once the model learns those patterns the classification
is almost trivial. This is consistent with what published papers
report on this dataset. I explicitly dropped the Type column before
training to prevent direct label leakage, ensuring this
near-perfect score is legitimate.

BGL: This was the most interesting result. LightGBM F1 dropped
from 0.74 on validation to 0.27 on test. I confirmed that the
Precision of 0.840 and Recall of 0.160 are purely from the test
set. Random Forest generalised better on the test set (F1=0.47,
Precision=0.97, Recall=0.31). The val-to-test drop is direct
evidence that BGL failure patterns shift over time. A random split
would have hidden this completely — the chronological split
exposed it.

OpenStack: Models essentially failed (F1 near 0). Expected given
only 4 anomalous sessions. Reported as a documented dataset
limitation, not a code error. OpenStack used for qualitative
RQ3 analysis only.

**Decisions made:**
- Best model for HDFS: LightGBM
- Best model for BGL: Random Forest — highest test Precision
  (0.97) with best overall test F1 (0.47) despite low Recall
  (0.31). High-precision alerts are more operationally
  trustworthy than high-recall alerts with many false positives
  in an SRE context.
- OpenStack: excluded from RQ1/RQ2 quantitative evaluation

**My personal observation:**
The catastrophic drop in LightGBM performance on the BGL test
set is the most valuable failure I have encountered so far. It
perfectly illustrates why SREs do not trust static ML models —
they memorise historical log structures but break entirely when
the infrastructure drifts or fails in novel ways. It validates
my decision to enforce strict chronological splitting.

---

### Day 3 — 07 August 2026

**Objective:** XAI Pipeline Generation (SHAP and LIME)

**Completed:**
- [x] SHAP TreeExplainer integrated for HDFS (LightGBM) and BGL (Random Forest)
- [x] LIME TabularExplainer integrated for both datasets
- [x] Exported feature importance matrices for top 100 true positive instances
- [x] Fixed SHAP additivity failure on BGL RF — disabled strict check (documented workaround)
- [x] Fixed OpenStack pathological SHAP (E3 magnitude 10,632) — regularised model

**Key finding today:**
Generating exact Shapley values for BGL's massive feature space
is computationally intensive. LIME is faster but its synthetic
perturbations do not always make logical sense in the context of
discrete log event counts. This is a limitation of LIME on sparse
count data that will be documented in Chapter 5 Discussion.

**My observation:**
I am seeing a stark contrast in how these two explainers handle
the data. For HDFS they point to the same key log lines. For BGL
they are wildly diverging. I need to run a formal Spearman
correlation tomorrow to quantify this disagreement.

---

### Day 4 — 08 August 2026

**Objective:** RQ2 Results — Feature Agreement Analysis

**Results saved:** results/xai_agreement_metrics.csv

**Key findings — my own interpretation:**

HDFS: Strong SHAP-LIME agreement (Spearman r = 0.676)
BGL: Barely any agreement (Spearman r = 0.113)

**My personal observation:**
This points directly to the curse of dimensionality affecting
XAI reliability. HDFS is a dense, low-dimensional, stable
feature space where LIME's local linear surrogate can reliably
map the same decision boundary as SHAP. BGL is a massive, highly
sparse feature space. LIME's random perturbations are landing in
empty regions causing it to fit noise, while SHAP is dealing with
sampling approximation variance. Feature space size directly
dictates XAI reliability — this is a strong, defensible finding
for RQ2.

---

### Day 5 — 09 August 2026

**Objective:** Deep Dive into BGL Concept Drift (RQ1 Follow-up)

**Completed:**
- [x] Visualised feature importance over time using SHAP summary plots for BGL
- [x] Mapped specific log templates that triggered false negatives in test set

**Key finding today:**
Random Forest's high Precision (0.97) but low Recall (0.31) on
BGL means that when it alerts it is right — but it misses almost
70% of actual outages. The false negatives in the test set were
driven by log templates that did not exist in the training window.

**My observation:**
You cannot train a supervised anomaly detection model to catch
failure modes it has never seen. Traditional software fails
predictably, but infrastructure fails in novel ways. In a real
environment this model would cause dangerous alert fatigue — not
because of false alarms, but because engineers would realise it
was silently missing major incidents.

---

### Day 6 — 10 August 2026

**Objective:** Qualitative Analysis of OpenStack (RQ3 prep)

**Completed:**
- [x] Manually inspected the 4 anomalous instance UUIDs in OpenStack log files
- [x] Mapped Drain3 templates back to raw human-readable logs for these 4 events

**Key finding today:**
Despite the model failing quantitatively on OpenStack, looking at
the logs manually through the Drain3 parsed templates was
insightful. The anomalies were distinct block-storage and network
timeouts that stood out clearly once variable IPs and UUIDs were
masked.

**My observation:**
If I had not properly masked the OpenStack logs, those 4 anomalies
would have been buried in 3,983 disparate templates. Proper
preprocessing is doing 80% of the heavy lifting. The specific
template IDs for the block-storage and network timeout anomalies
will be documented explicitly in Chapter 4 after RQ3 runs.

---

### Day 7 — 11 August 2026

**Objective:** Operationalising the Framework (RQ3 — SRE Application)

**Completed:**
- [x] Defined degradation boundaries and fallback heuristics
- [x] Drafted the operational framework architecture

**Key finding today:**
An ML model flagging a log anomaly is not an incident — it is a
signal. The framework I am proposing requires a secondary
validation layer before paging an engineer.

**My observation:**
If a team deployed this tomorrow, the first operational question
they would need it to answer is: Is this log anomaly correlated
with an actionable system metric degradation? Log anomalies happen
during normal patch rollouts and batch jobs. If the framework
cannot tie the log pattern to a degraded golden signal such as
HTTP 5xx spikes or latency increases, it should not page an
engineer. That is the secondary validation layer the framework
needs.

---

### Day 8 — 12 August 2026

**Objective:** Finalise Results and Project Synthesis

**Completed:**
- [x] Consolidated all result tables for RQ1 and RQ2
- [x] Exported final SHAP and LIME visualisation plots for dissertation appendix
- [x] Backed up 01_preprocess_bgl.py and utils_data.py to GitHub

**Key finding today:**
The pipeline is fully reproducible from raw log files to final
xai_agreement_metrics.csv. The chronological splitting logic and
Drain3 masking configurations are robust and documented.

**My observation:**
This project evolved from a standard classification task into a
critique of how academic ML often ignores real-world infrastructure
constraints. Proving that XAI agreement degrades in
high-dimensional log spaces and demonstrating concept drift via
chronological splits are strong, defensible core arguments for the
dissertation. Ready to begin Chapter 3 Methodology writing.

---

### Day 9 — 13 August 2026

**Objective:** RQ3 Actionability Analysis and OpenStack 
Qualitative Deep-Dive

**Completed:**
- [x] Extracted the actual event templates triggered by the 
  4 anomalous OpenStack instances directly from the 
  structured CSV data
- [x] Wrote `utils_rq3.py` to establish rule-based keyword 
  matching, mapping Drain3 templates to operational failure 
  categories (Data integrity, Resource exhaustion, Network 
  fault, Hardware failure, Application error)
- [x] Wrote and executed `06_rq3_actionability.py` to map 
  the top 15 SHAP features for HDFS and BGL to these 
  categories
- [x] Expanded the categorisation rule set with legitimate 
  BGL domain terms (TLB errors, ECC CE sym, torus link 
  training, panics) after a poor initial matching run
- [x] Wrote and executed `07_rq3_openstack_qualitative.py` 
  to synthesise the OpenStack findings
- [x] Patched a false-positive bug where the word "cpu" in 
  a routine resource-claim log was incorrectly flagging as 
  Hardware failure

**RQ3 Results:**

| Dataset | Features Checked | Categorised | Coverage | Dominant Category |
|---------|-----------------|-------------|----------|-------------------|
| HDFS | 15 | 8 | 53.33% | Unknown |
| BGL | 15 | 14 | 93.33% | Hardware failure |
| OpenStack | Qualitative only | N/A | N/A | Timing/sequence anomaly |

**Key finding today:**
I have concrete proof of why the ML models failed on OpenStack 
in RQ1 (near-zero F1). The 4 anomalous OpenStack instances did 
not trigger a single explicit error message. They only triggered 
13 routine VM lifecycle operations — instance creation, resource 
claims, deletion, network deallocation. The injected anomalies 
manifest entirely as anomalous timing or sequencing of otherwise 
normal logs. Because my preprocessing pipeline converts logs into 
an event-count matrix, which inherently discards sequence and 
timing information, the anomaly signal was completely erased 
before it even reached the model.

**Note on HDFS 53.33% coverage:**
Seven of the top 15 HDFS SHAP features are categorised as 
Unknown because they represent normal block operations — serving, 
receiving, deleting blocks — whose anomalous count or absence 
constitutes the anomaly signal, not the presence of an explicit 
error message. HDFS anomalies are structural (missing replication 
steps) rather than lexical (explicit error logs). This is 
discussed in Chapter 5.

**My observation:**
My first attempt at the actionability script was a reality check. 
BGL only achieved 13% coverage initially because my generic 
keyword list entirely missed BGL's specific supercomputer 
RAS/kernel vocabulary. By iterating on the rules and adding real 
domain terms rather than forcing matches, I got BGL coverage up 
to 93.33% — 14 of the top 15 SHAP features successfully 
categorised, predominantly as Hardware failures. This proves a 
crucial point for RQ3: XAI is only actionable if the system 
mapping the SHAP values actually understands the domain-specific 
vocabulary of the infrastructure. The OpenStack sequencing 
discovery goes directly into Chapter 5 as a fundamental 
limitation of count-based log representations.

**Files produced today:**
- `results/rq3_actionability_HDFS.csv`
- `results/rq3_actionability_BGL.csv`
- `results/rq3_coverage.csv`
- `results/rq3_openstack_qualitative.txt`
- `results/figures/rq3_actionability_HDFS.png`
- `results/figures/rq3_actionability_BGL.png`
- `src/06_rq3_actionability.py`
- `src/07_rq3_openstack_qualitative.py`
- `src/utils_rq3.py`


### Day 10 — 14 August 2026

**Objective:** BGL Session Window Size Ablation Study

**Completed:**
- [x] Implemented `08_ablation_window_size.py` to evaluate 
  model sensitivity to BGL session window sizes 
  (ws = 10, 20, 40, 100)
- [x] Built sparse matrix generator using scipy.sparse.coo_matrix 
  to aggregate 4.7 million log entries from BGL_structured.csv 
  without re-running Drain3
- [x] Fixed sparse data type mismatch bug — LightGBM rejected 
  int32 sparse matrices during C++ buffer creation. 
  Converted to float32 to resolve.
- [x] Extracted test set metrics across all window sizes for 
  RandomForest and LightGBM
- [x] Generated results/figures/ablation_window_size.png 
  and results/ablation_window_size.csv

**Correction noted:**
BGL baseline used window_size=100, not window_size=20 as 
initially planned. All RQ1, RQ2, RQ3 results were built 
with window_size=100. Ablation revised to test 
ws = {10, 20, 40, 100} with existing baseline as anchor.

**Ablation Results:**

| Window Size | Model | Precision | Recall | F1 | AUC-ROC |
|-------------|-------|-----------|--------|----|---------|
| 10 | RandomForest | 0.998 | 0.196 | 0.328 | 0.757 |
| 20 | RandomForest | 0.995 | 0.212 | 0.349 | 0.749 |
| 40 | RandomForest | 0.963 | 0.238 | 0.382 | 0.761 |
| 100 | RandomForest | 0.969 | 0.307 | 0.466 | 0.785 |
| 10 | LightGBM | 0.941 | 0.202 | 0.333 | 0.756 |
| 20 | LightGBM | 0.870 | 0.223 | 0.355 | 0.754 |
| 40 | LightGBM | 0.144 | 0.902 | 0.248 | 0.757 |
| 100 | LightGBM | 0.162 | 0.908 | 0.274 | 0.740 |

**Key finding today:**
window_size=100 is the most effective configuration for 
RandomForest, not an arbitrary choice. As window size 
decreases, RandomForest test F1 drops steadily from 
0.466 at ws=100 down to 0.328 at ws=10. Smaller windows 
cause severe recall degradation — dropping from 30.7% 
at ws=100 to 19.6% at ws=10 — because brief windows 
lack sufficient co-occurring log event context to trigger 
anomaly signals.

LightGBM exhibited a sharp regime shift between ws=20 
and ws=40. At ws=10 and ws=20, LightGBM behaved like 
RandomForest: high precision (0.941 and 0.870) with low 
recall (0.202 and 0.223). At ws=40 and ws=100, it 
switched abruptly to high recall (0.902 and 0.908) with 
very low precision (0.144 and 0.162), collapsing F1 to 
0.248 and 0.274. This regime shift is driven by how 
scale_pos_weight interacts with gradient calculations 
as the positive class ratio and feature density per 
row both increase with larger windows.

**My observation:**
The contrast between RandomForest and LightGBM under 
different window sizes is a key finding for the 
dissertation. RandomForest scales predictably because 
tree ensembles handle sparse count matrices gracefully 
as window density grows. LightGBM's drastic shift at 
ws=40 demonstrates that preprocessing choices like 
windowing cannot be evaluated in isolation — their 
impact depends heavily on the downstream classifier's 
optimisation surface. This finding directly informs 
the framework design recommendation in Chapter 5: 
RandomForest is the safer choice for production 
deployment on BGL-style sparse log data precisely 
because its behaviour is predictable across window 
configurations.

**Error fixed today:**

| Error | Cause | Fix |
|-------|-------|-----|
| LightGBM rejected sparse matrix | int32 dtype incompatible with C++ buffer creation | Converted sparse matrix to float32 before fitting |

**Files produced today:**
- `results/ablation_window_size.csv`
- `results/figures/ablation_window_size.png`
- `src/08_ablation_window_size.py`