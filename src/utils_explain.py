"""Shared helpers for RQ2 explainability analysis (SHAP, LIME, faithfulness).

Used by 04_explain_rq2.py (HDFS, BGL) and 05_explain_openstack_shap.py.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def stratified_sample(X, y, n_samples, random_state=42):
    """Sample n_samples rows from (X, y), preserving class proportions.

    Guarantees at least 1 positive-class row is included (when any exist),
    even for extremely rare classes, using np.ceil rather than round/floor
    for the positive count. This matters here because rare-class rows are
    exactly what a waterfall plot and faithfulness test need to see.
    """
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    pos_frac = len(pos_idx) / len(y)
    n_pos = min(len(pos_idx), max(1, int(np.ceil(n_samples * pos_frac))))
    n_neg = min(len(neg_idx), n_samples - n_pos)

    rng = np.random.RandomState(random_state)
    pos_sample = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)

    idx = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def extract_positive_class_shap(explanation):
    """Extract the positive-class SHAP values array from a shap.Explanation object.

    RandomForestClassifier (a predict_proba-based model) yields shape
    (n, features, 2) — one array per class; LightGBM's booster yields a
    single (n, features) array already in terms of the positive class's
    log-odds. Both are normalised here to (n, features).
    """
    values = np.array(explanation.values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def get_lime_feature_weights(explanation, feature_cols):
    """Convert a LIME explanation's as_list() output into a full feature_name -> weight dict.

    With discretize_continuous=False, LIME's condition strings are of the
    form "FeatureName <= value" / "FeatureName > value" / plain
    "FeatureName", so the base feature name is recovered by matching
    against the known feature_cols rather than fragile string splitting.
    """
    weights = {f: 0.0 for f in feature_cols}
    feature_set = set(feature_cols)
    for condition, weight in explanation.as_list():
        matched = None
        for token in re.split(r"\s|<=|>=|<|>", condition):
            if token in feature_set:
                matched = token
                break
        if matched is not None:
            weights[matched] = weight
    return weights


def faithfulness_test(model, X_sample, importance_matrix, feature_cols, k_values, method_name):
    """Measure faithfulness via a deletion/comprehensiveness test.

    For each instance and each k, the top-k features by |importance| are
    zeroed out (0 = "event never occurred", a meaningful absence baseline
    for event-count features, not an arbitrary imputation), and the drop in
    predicted anomaly probability is recorded. A faithful explanation
    should show larger probability drops for larger k, since it is
    correctly identifying the features the model actually relies on.
    """
    baseline_proba = model.predict_proba(X_sample)[:, 1]
    rows = []
    for k in k_values:
        drops = []
        for i in range(len(X_sample)):
            top_k_idx = np.argsort(-np.abs(importance_matrix[i]))[:k]
            perturbed = X_sample.iloc[[i]].copy()
            perturbed.iloc[0, top_k_idx] = 0
            new_proba = model.predict_proba(perturbed)[:, 1][0]
            drops.append(baseline_proba[i] - new_proba)
        rows.append({
            "Method": method_name,
            "k": k,
            "mean_prob_drop": float(np.mean(drops)),
            "std_prob_drop": float(np.std(drops)),
        })
    return rows


def random_baseline_faithfulness(model, X_sample, feature_cols, k_values, random_state=42):
    """Faithfulness control: drop k random features instead of top-ranked ones.

    Establishes whether SHAP/LIME rankings actually beat chance — if random
    deletion causes similar probability drops, the "faithful" ranking isn't
    adding real explanatory value.
    """
    rng = np.random.RandomState(random_state)
    n_features = len(feature_cols)
    random_importance = rng.rand(len(X_sample), n_features)
    return faithfulness_test(model, X_sample, random_importance, feature_cols, k_values, "Random")


def compute_spearman_agreement(shap_values, lime_weights_df, feature_cols):
    """Compute Spearman rank correlation between SHAP and LIME global feature importances.

    Global importance per method = mean absolute value across the sampled
    instances. Comparing ranks (not raw magnitudes) is appropriate since
    SHAP values and LIME coefficients live on different scales.
    """
    shap_importance = np.abs(shap_values).mean(axis=0)
    lime_importance = lime_weights_df[feature_cols].abs().mean(axis=0).values
    corr, pvalue = spearmanr(shap_importance, lime_importance)
    return corr, pvalue, shap_importance, lime_importance
