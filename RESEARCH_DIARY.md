# Research Diary

**Candidate:** Mintu Ghosh
**Title:** Explainable AI for Cloud Incident Prediction 
           using OpenTelemetry
**Programme:** MSc Machine Learning and Artificial 
               Intelligence — LJMU

---

## Purpose of This Diary

This diary records my personal reflections as a 
researcher — what I learned, what surprised me, 
what frustrated me, and how my thinking evolved 
throughout the project. It is separate from the 
technical log.

---

## Week 1 — 05 to 11 August 2026

**What surprised me most:**
When I discovered OpenStack only had 4 anomalous 
sessions, my first reaction was frustration. Then I 
realised this was actually an important finding — it 
showed me that real-world datasets are far messier 
than academic papers make them sound. No paper I read 
mentioned this specific limitation with the LogPai 
OpenStack dataset.

**What I learned about my own assumptions:**
I assumed SHAP and LIME would broadly agree on the 
most important features. The BGL result (Spearman 
r=0.113) completely challenged that assumption. I 
had to rethink what "explainability" actually means 
when two legitimate methods produce nearly uncorrelated 
feature rankings on the same model.

**Connection to my SRE experience:**
The BGL val-to-test F1 drop (0.74 to 0.27) reminded 
me of every time a model that looked great in staging 
fell apart in production. I have seen this pattern 
dozens of times in my career. The chronological 
split exposed exactly the kind of drift that random 
splits hide — and that production environments never 
forgive.

**What I would do differently:**
If I had more time, I would have investigated 
OpenStack's timing-based anomalies more deeply. The 
finding that anomalies manifest as sequencing 
anomalies rather than lexical errors is something 
worth pursuing in a follow-up study.

**Supervisor interaction:**
[Add notes from supervisor meetings here]

---

## Week 2 — 12 to 18 August 2026

**What surprised me most:**
Writing the chapter outlines before any prose exposed a gap I would
not have caught otherwise: I could not actually point to a specific
figure or CSV backing every planned sentence about BGL's SHAP
stability. Chasing that gap down meant opening the actual SHAP summary
plot image for the first time since it was generated days earlier —
and the x-axis went up to 5e11. A model whose output is a probability
between 0 and 1 cannot honestly produce SHAP values in the hundreds of
billions. That was not a plotting quirk; it was a corrupted
computation that had been sitting in my results directory, feeding
three separate result files, without me noticing.

**What I learned about my own assumptions:**
I had written, in my own log, that a SHAP additivity-check failure was
"a documented SHAP/sklearn interaction, not a sign of incorrect SHAP
values." I believed that because the code ran without crashing once I
disabled the check. That is exactly the failure mode I would flag
immediately in a code review from someone else: treating "no exception
raised" as equivalent to "the computation is correct." I did not apply
my own standard to my own pipeline until a visual inspection forced the
issue. The fix — feature_perturbation='interventional' with a real
background sample — was in SHAP's own error message the entire time.

**Connection to my SRE experience:**
This is the log-monitoring equivalent of a service that silences a
health-check alert instead of fixing what the health check was
actually detecting. I have seen teams do exactly this under deadline
pressure — disable the alert, keep shipping, and only discover months
later that the metric it was protecting had been silently wrong the
whole time. Catching it here, in a controlled research pipeline with
full reproducibility logging, is the version of this failure I can
actually afford. In production it would have shipped.

**Open questions I want to explore:**
The corrected finding reverses my own earlier conclusion: BGL's SHAP
feature rankings are actually fairly stable across window sizes
(Jaccard 0.43-0.667), and the real instability is between SHAP and
LIME as methods (Spearman r=0.138), not preprocessing choice. I want
to understand why — is LIME's local linear surrogate fundamentally
worse-suited to BGL's ~1000-dimensional sparse feature space than
SHAP's tree-structure-aware computation, or is this an artifact of
LIME's own sampling budget that a larger num_samples would close? I
do not have time to chase this further in this dissertation, but it
feels like the actual research question underneath RQ2, not just an
implementation detail of it.

---

## Week 3 — 19 to 25 August 2026

[Fill when reached]

---

## Week 4 — 26 August to 01 September 2026

[Fill when reached]

---

## What This Research Changed in My Thinking

[Write this last — after the dissertation is complete.
What do you now believe about XAI and AIOps that you 
did not believe before you started?]