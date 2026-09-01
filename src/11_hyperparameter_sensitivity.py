"""Hyperparameter sensitivity study: LightGBM on BGL (RQ1 follow-up).

Grid search over n_estimators x learning_rate x num_leaves (27
combinations), each trained on BGL's existing chronological train split and
evaluated on the existing chronological test split (no re-splitting, no
val-set tuning — this deliberately measures test-set sensitivity to
hyperparameters, not a proper tuning procedure). scale_pos_weight is held
fixed at the train split's actual imbalance ratio (not swept) throughout,
consistent with every other LightGBM run in this project.
"""

import itertools
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score

from utils_data import get_feature_columns

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

N_ESTIMATORS_GRID = [100, 200, 500]
LEARNING_RATE_GRID = [0.01, 0.05, 0.1]
NUM_LEAVES_GRID = [15, 31, 63]
RANDOM_STATE = 42


def load_bgl_splits():
    """Load BGL's existing chronological train/test splits."""
    splits_dir = os.path.join(PROCESSED_DIR, "BGL", "splits")
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    feature_cols = get_feature_columns(train_df)
    return (
        train_df[feature_cols], train_df["Label"],
        test_df[feature_cols], test_df["Label"],
    )


def run_grid_search(X_train, y_train, X_test, y_test):
    """Train and evaluate LightGBM across all 27 hyperparameter combinations."""
    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight fixed at {scale_pos_weight:.2f} (train imbalance ratio) across all combinations")

    combos = list(itertools.product(N_ESTIMATORS_GRID, LEARNING_RATE_GRID, NUM_LEAVES_GRID))
    print(f"  Running {len(combos)} combinations...")

    rows = []
    for i, (n_estimators, learning_rate, num_leaves) in enumerate(combos, start=1):
        model = LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, verbosity=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_auc_roc = roc_auc_score(y_test, y_proba)

        print(f"    [{i:>2}/{len(combos)}] n_estimators={n_estimators}, lr={learning_rate}, "
              f"num_leaves={num_leaves} -> F1={test_f1:.4f}, AUC-ROC={test_auc_roc:.4f}")
        rows.append({
            "n_estimators": n_estimators, "learning_rate": learning_rate, "num_leaves": num_leaves,
            "test_f1": test_f1, "test_auc_roc": test_auc_roc,
        })
    return pd.DataFrame(rows)


def analyze_impact(results_df):
    """Report the best combination, F1 range, and which parameter drives F1 the most.

    "Most impactful" is defined as the parameter whose marginal mean F1
    (averaged over the other two parameters) has the largest range across
    its own grid values — i.e. the parameter whose main effect swings F1
    the most, holding the grid design fixed.
    """
    best_row = results_df.loc[results_df["test_f1"].idxmax()]
    f1_range = results_df["test_f1"].max() - results_df["test_f1"].min()

    print("\n  Best combination by test F1:")
    print(f"    n_estimators={int(best_row['n_estimators'])}, "
          f"learning_rate={best_row['learning_rate']}, num_leaves={int(best_row['num_leaves'])} "
          f"-> F1={best_row['test_f1']:.4f}, AUC-ROC={best_row['test_auc_roc']:.4f}")

    print(f"\n  Range of F1 across all 27 combinations: {f1_range:.4f} "
          f"(max={results_df['test_f1'].max():.4f}, min={results_df['test_f1'].min():.4f})")

    print("\n  Main-effect F1 range per parameter (marginal mean, averaged over the other two):")
    param_ranges = {}
    for param in ["n_estimators", "learning_rate", "num_leaves"]:
        marginal_means = results_df.groupby(param)["test_f1"].mean()
        param_range = marginal_means.max() - marginal_means.min()
        param_ranges[param] = param_range
        print(f"    {param}: range={param_range:.4f}  (means per value: {marginal_means.to_dict()})")

    most_impactful = max(param_ranges, key=param_ranges.get)
    print(f"\n  Most impactful parameter on F1: {most_impactful} "
          f"(main-effect range={param_ranges[most_impactful]:.4f})")

    return best_row, f1_range, most_impactful


def make_heatmap(results_df):
    """Save a 3-panel heatmap (one per num_leaves) of n_estimators x learning_rate, colour=test_f1."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    vmin, vmax = results_df["test_f1"].min(), results_df["test_f1"].max()

    fig, axes = plt.subplots(1, len(NUM_LEAVES_GRID), figsize=(15, 5), sharey=True)
    for ax, num_leaves in zip(axes, NUM_LEAVES_GRID):
        panel = results_df[results_df["num_leaves"] == num_leaves]
        pivot = panel.pivot(index="n_estimators", columns="learning_rate", values="test_f1")
        pivot = pivot.reindex(index=N_ESTIMATORS_GRID, columns=LEARNING_RATE_GRID)

        im = ax.imshow(pivot.values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(LEARNING_RATE_GRID)))
        ax.set_xticklabels(LEARNING_RATE_GRID)
        ax.set_yticks(range(len(N_ESTIMATORS_GRID)))
        ax.set_yticklabels(N_ESTIMATORS_GRID)
        ax.set_xlabel("learning_rate")
        ax.set_title(f"num_leaves={num_leaves}")
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                ax.text(c, r, f"{pivot.values[r, c]:.3f}", ha="center", va="center",
                        color="white", fontsize=9)
    axes[0].set_ylabel("n_estimators")

    fig.colorbar(im, ax=axes, label="Test F1", fraction=0.03, pad=0.02)
    fig.suptitle("LightGBM on BGL: Test F1 Hyperparameter Sensitivity")
    fig_path = os.path.join(FIGURES_DIR, "hyperparameter_heatmap.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved figure: {fig_path}")


def main():
    """Run the 27-combination LightGBM hyperparameter grid search on BGL and report sensitivity."""
    print("STEP: LightGBM hyperparameter sensitivity study on BGL")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        X_train, y_train, X_test, y_test = load_bgl_splits()
        print(f"  train={len(X_train):,} ({int(y_train.sum())} anomalous), "
              f"test={len(X_test):,} ({int(y_test.sum())} anomalous)")

        results_df = run_grid_search(X_train, y_train, X_test, y_test)

        out_path = os.path.join(RESULTS_DIR, "hyperparameter_sensitivity.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")

        analyze_impact(results_df)
        make_heatmap(results_df)

        print("\nSTEP COMPLETE: Hyperparameter sensitivity study finished successfully")

    except Exception:
        print("\nERROR during hyperparameter sensitivity study:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
