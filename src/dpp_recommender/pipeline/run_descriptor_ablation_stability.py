from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from dpp_recommender.meta_dataset.meta_dataset import DPP_FEATURE_COLUMNS, _variant_label_frame
from dpp_recommender.pipeline.run_descriptor_ablation import (
    EXPECTED_VARIANT_COUNT,
    FEATURE_CONFIGURATIONS,
    META_DATASET_PATH,
    N_REPEATS,
    N_SPLITS,
    RANDOM_SEED,
    REPO_ROOT,
    _tree_constructor,
)


OUTPUT_JSON_PATH = REPO_ROOT / "outputs" / "experiments" / "descriptor_ablation_stability.json"
OUTPUT_CSV_PATH = REPO_ROOT / "outputs" / "experiments" / "descriptor_ablation_stability.csv"
BOOTSTRAP_RESAMPLES = 1000


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
    }


def _bootstrap_ci(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    bootstrap_means = array[indices].mean(axis=1)
    return {
        "lower_95": float(np.percentile(bootstrap_means, 2.5)),
        "upper_95": float(np.percentile(bootstrap_means, 97.5)),
    }


def _paired_summary(ablated: list[float], full: list[float]) -> dict:
    deltas = (np.asarray(ablated) - np.asarray(full)).tolist()
    bootstrap_ci = _bootstrap_ci(deltas)
    return {
        "paired_differences": deltas,
        "distribution": _summary(deltas),
        "bootstrap_ci_95": bootstrap_ci,
        "mean": float(np.mean(deltas)),
        "std": float(np.std(deltas)),
        "median": float(np.median(deltas)),
        "minimum": float(np.min(deltas)),
        "maximum": float(np.max(deltas)),
        "p25": float(np.percentile(deltas, 25)),
        "p75": float(np.percentile(deltas, 75)),
        "p95": float(np.percentile(deltas, 95)),
        "ci_excludes_zero": bool(
            bootstrap_ci["lower_95"] > 0 or bootstrap_ci["upper_95"] < 0
        ),
        "wins": int(sum(delta > 0 for delta in deltas)),
        "losses": int(sum(delta < 0 for delta in deltas)),
        "ties": int(sum(delta == 0 for delta in deltas)),
    }


def _run_fold_metrics(labels: pd.DataFrame, feature_columns: list[str], splits: list[tuple]) -> list[dict]:
    X = labels[feature_columns].fillna(0.0)
    y = labels["best_strategy"]
    label_order = sorted(y.unique().tolist())
    metrics = []
    for fold_index, (train_indices, test_indices) in enumerate(splits):
        model = _tree_constructor()
        model.fit(X.iloc[train_indices], y.iloc[train_indices])
        predictions = model.predict(X.iloc[test_indices])
        y_test = y.iloc[test_indices]
        metrics.append({
            "fold_index": fold_index,
            "repetition": fold_index // N_SPLITS,
            "fold_within_repetition": fold_index % N_SPLITS,
            "macro_f1": float(f1_score(y_test, predictions, labels=label_order, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        })
    return metrics


def run_stability_analysis(meta_dataset_path: Path = META_DATASET_PATH) -> dict:
    meta_dataset = pd.read_csv(meta_dataset_path)
    labels = _variant_label_frame(meta_dataset)
    if len(labels) != EXPECTED_VARIANT_COUNT:
        raise ValueError(f"Expected {EXPECTED_VARIANT_COUNT} labelled variants, got {len(labels)}")

    X = labels[DPP_FEATURE_COLUMNS].fillna(0.0)
    y = labels["best_strategy"]
    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    splits = list(splitter.split(X, y))
    fold_metrics = {
        name: _run_fold_metrics(labels, feature_columns, splits)
        for name, feature_columns in FEATURE_CONFIGURATIONS.items()
    }
    full_metrics = fold_metrics["full_dpp"]

    configurations = {}
    for name, metrics in fold_metrics.items():
        macro_f1 = [row["macro_f1"] for row in metrics]
        balanced_accuracy = [row["balanced_accuracy"] for row in metrics]
        result = {
            "fold_metrics": metrics,
            "macro_f1": _summary(macro_f1),
            "balanced_accuracy": _summary(balanced_accuracy),
        }
        if name != "full_dpp":
            result["paired_vs_full_dpp"] = {
                "macro_f1": _paired_summary(macro_f1, [row["macro_f1"] for row in full_metrics]),
                "balanced_accuracy": _paired_summary(
                    balanced_accuracy, [row["balanced_accuracy"] for row in full_metrics]
                ),
            }
        configurations[name] = result

    return {
        "dataset": {
            "source": str(meta_dataset_path.relative_to(REPO_ROOT)),
            "labelled_variant_count": len(labels),
            "descriptor_names": DPP_FEATURE_COLUMNS,
        },
        "feature_configurations": FEATURE_CONFIGURATIONS,
        "model": {"name": "DecisionTreeClassifier", "max_depth": 3, "random_state": RANDOM_SEED},
        "validation": {
            "method": "RepeatedStratifiedKFold",
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "random_state": RANDOM_SEED,
            "number_of_evaluations": len(splits),
            "identical_splits_across_configurations": True,
        },
        "paired_bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "random_state": RANDOM_SEED,
            "confidence_level": 0.95,
            "method": "percentile bootstrap of the mean paired fold difference",
        },
        "configurations": configurations,
    }


def _write_csv(summary: dict) -> None:
    rows = []
    for name, result in summary["configurations"].items():
        paired = result.get("paired_vs_full_dpp", {})
        macro_delta = paired.get("macro_f1", {})
        bacc_delta = paired.get("balanced_accuracy", {})
        for fold in result["fold_metrics"]:
            rows.append({
                "configuration": name,
                **fold,
                "delta_macro_f1_vs_full": (
                    fold["macro_f1"] - summary["configurations"]["full_dpp"]["fold_metrics"][fold["fold_index"]]["macro_f1"]
                    if name != "full_dpp" else 0.0
                ),
                "delta_balanced_accuracy_vs_full": (
                    fold["balanced_accuracy"] - summary["configurations"]["full_dpp"]["fold_metrics"][fold["fold_index"]]["balanced_accuracy"]
                    if name != "full_dpp" else 0.0
                ),
                "paired_delta_mean": macro_delta.get("distribution", {}).get("mean"),
                "paired_delta_ci_lower_95": macro_delta.get("bootstrap_ci_95", {}).get("lower_95"),
                "paired_delta_ci_upper_95": macro_delta.get("bootstrap_ci_95", {}).get("upper_95"),
                "paired_bacc_delta_mean": bacc_delta.get("distribution", {}).get("mean"),
                "paired_bacc_ci_lower_95": bacc_delta.get("bootstrap_ci_95", {}).get("lower_95"),
                "paired_bacc_ci_upper_95": bacc_delta.get("bootstrap_ci_95", {}).get("upper_95"),
            })
    pd.DataFrame(rows).to_csv(OUTPUT_CSV_PATH, index=False)


def main() -> None:
    summary = run_stability_analysis()
    OUTPUT_JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(summary)
    print(f"JSON saved to {OUTPUT_JSON_PATH}")
    print(f"CSV saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()