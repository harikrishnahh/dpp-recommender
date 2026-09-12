from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from dpp_recommender.meta_dataset.meta_dataset import DPP_FEATURE_COLUMNS, _variant_label_frame
from dpp_recommender.pipeline.run_final_validation import _run_repeated_cv


REPO_ROOT = Path(__file__).resolve().parents[3]
META_DATASET_PATH = REPO_ROOT / "outputs" / "experiments" / "robust_meta_dataset_10_stations.csv"
OUTPUT_JSON_PATH = REPO_ROOT / "outputs" / "experiments" / "descriptor_ablation_results.json"
OUTPUT_CSV_PATH = REPO_ROOT / "outputs" / "experiments" / "descriptor_ablation_results.csv"
RANDOM_SEED = 42
N_SPLITS = 5
N_REPEATS = 5
EXPECTED_VARIANT_COUNT = 144


FEATURE_CONFIGURATIONS = {
    "full_dpp": DPP_FEATURE_COLUMNS.copy(),
    "without_missing_ratio": [
        "mean_gap_length", "mean_lag1_autocorrelation", "mean_trend_strength",
        "mean_absolute_pairwise_correlation",
    ],
    "without_mean_gap_length": [
        "missing_ratio", "mean_lag1_autocorrelation", "mean_trend_strength",
        "mean_absolute_pairwise_correlation",
    ],
    "without_mean_lag1_autocorrelation": [
        "missing_ratio", "mean_gap_length", "mean_trend_strength",
        "mean_absolute_pairwise_correlation",
    ],
    "without_mean_trend_strength": [
        "missing_ratio", "mean_gap_length", "mean_lag1_autocorrelation",
        "mean_absolute_pairwise_correlation",
    ],
    "without_mean_absolute_pairwise_correlation": [
        "missing_ratio", "mean_gap_length", "mean_lag1_autocorrelation",
        "mean_trend_strength",
    ],
}


def _tree_constructor() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED)


def _metric_summary(result: dict) -> dict:
    return {"macro_f1": result["macro_f1"], "balanced_accuracy": result["balanced_accuracy"]}


def run_ablation(meta_dataset_path: Path = META_DATASET_PATH) -> dict:
    meta_dataset = pd.read_csv(meta_dataset_path)
    labels = _variant_label_frame(meta_dataset)
    if len(labels) != EXPECTED_VARIANT_COUNT:
        raise ValueError(f"Expected {EXPECTED_VARIANT_COUNT} labelled variants, got {len(labels)}")

    results = {
        "majority": _metric_summary(_run_repeated_cv(
            labels, None, DPP_FEATURE_COLUMNS, n_splits=N_SPLITS, n_repeats=N_REPEATS
        )),
        "missing_ratio_only": _metric_summary(_run_repeated_cv(
            labels, _tree_constructor, ["missing_ratio"], n_splits=N_SPLITS, n_repeats=N_REPEATS
        )),
    }
    for name, feature_columns in FEATURE_CONFIGURATIONS.items():
        results[name] = _metric_summary(_run_repeated_cv(
            labels, _tree_constructor, feature_columns, n_splits=N_SPLITS, n_repeats=N_REPEATS
        ))

    return {
        "dataset": {
            "source": str(meta_dataset_path.relative_to(REPO_ROOT)),
            "labelled_variant_count": len(labels),
            "descriptor_names": DPP_FEATURE_COLUMNS,
        },
        "validation": {
            "method": "RepeatedStratifiedKFold",
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "random_state": RANDOM_SEED,
            "metric_aggregation": "mean and population standard deviation across 25 folds",
        },
        "model": {"name": "DecisionTreeClassifier", "max_depth": 3, "random_state": RANDOM_SEED},
        "feature_configurations": {
            **FEATURE_CONFIGURATIONS,
            "missing_ratio_only": ["missing_ratio"],
            "majority": [],
        },
        "results": results,
    }


def _write_csv(summary: dict) -> None:
    rows = []
    for configuration, result in summary["results"].items():
        rows.append({
            "configuration": configuration,
            "macro_f1_mean": result["macro_f1"]["mean"],
            "macro_f1_sd": result["macro_f1"]["std"],
            "balanced_accuracy_mean": result["balanced_accuracy"]["mean"],
            "balanced_accuracy_sd": result["balanced_accuracy"]["std"],
        })
    pd.DataFrame(rows).to_csv(OUTPUT_CSV_PATH, index=False)


def main() -> None:
    summary = run_ablation()
    OUTPUT_JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(summary)
    for configuration, result in summary["results"].items():
        print(
            f"{configuration}: Macro-F1 {result['macro_f1']['mean']:.4f} +/- {result['macro_f1']['std']:.4f}; "
            f"Balanced Accuracy {result['balanced_accuracy']['mean']:.4f} +/- "
            f"{result['balanced_accuracy']['std']:.4f}"
        )
    print(f"JSON saved to {OUTPUT_JSON_PATH}")
    print(f"CSV saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()