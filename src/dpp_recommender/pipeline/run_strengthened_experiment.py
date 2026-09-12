from __future__ import annotations

import hashlib
import json
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from dpp_recommender.dataset.ingest import load_french_piezo_station
from dpp_recommender.descriptor_extraction import extract_dpp
from dpp_recommender.meta_dataset.meta_dataset import DPP_FEATURE_COLUMNS
from dpp_recommender.strategy_evaluation import (
    DEFAULT_STRATEGY_NAMES,
    evaluate_strategy_benchmark,
)
from dpp_recommender.variant_generator.generator import generate_variants


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPO_ROOT / "data" / "raw" / "dataset_2015_2021.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "experiments"
CACHE_DIR = OUTPUT_DIR / "strengthened_station_cache"
CACHE_VERSION = "v3_causal_test_preprocessing"
RANDOM_SEED = 42
MISSINGNESS_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
VARIANTS_PER_LEVEL = 8
BENCHMARK_FOLDS = 3
N_SPLITS = 5
N_REPEATS = 5
MIN_OBSERVATIONS = 2000
MIN_SPAN_DAYS = 2000
MIN_PREDICTOR_COVERAGE = 0.95
MIN_TARGET_COVERAGE = 1.0
PREDICTORS = ["tp", "e"]
TARGET = "p"

CONVENTIONAL_FEATURE_COLUMNS = [
    "observation_count",
    "predictor_count",
    "missing_cell_count",
    "missing_ratio",
    "missing_ratio_sd_by_predictor",
    "predictor_std_mean",
    "predictor_skew_mean",
    "predictor_lag1_autocorrelation_mean",
    "predictor_pairwise_correlation_abs",
]
REPRESENTATIONS = {
    "missing_ratio_only": ["missing_ratio"],
    "full_dpp": DPP_FEATURE_COLUMNS.copy(),
    "conventional_meta_features": CONVENTIONAL_FEATURE_COLUMNS.copy(),
}


def _stable_hash(values: list[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def screen_stations(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    records = []
    for station_id, group in raw.groupby("bss", dropna=True):
        valid_time = group.dropna(subset=["time"])
        span_days = int((valid_time["time"].max() - valid_time["time"].min()).days) if not valid_time.empty else 0
        checks = {
            "minimum_observations": bool(len(valid_time) >= MIN_OBSERVATIONS),
            "minimum_span_days": bool(span_days >= MIN_SPAN_DAYS),
            "predictor_tp_coverage": bool(group["tp"].notna().mean() >= MIN_PREDICTOR_COVERAGE),
            "predictor_e_coverage": bool(group["e"].notna().mean() >= MIN_PREDICTOR_COVERAGE),
            "target_p_complete": bool(group["p"].notna().mean() >= MIN_TARGET_COVERAGE),
            "chronological_fold_geometry": bool(len(valid_time) >= BENCHMARK_FOLDS + 2),
        }
        reasons = [name for name, passed in checks.items() if not passed]
        records.append({
            "station_id": str(station_id),
            "raw_rows": int(len(group)),
            "valid_time_rows": int(len(valid_time)),
            "span_days": span_days,
            "tp_coverage": float(group["tp"].notna().mean()),
            "e_coverage": float(group["e"].notna().mean()),
            "p_coverage": float(group["p"].notna().mean()),
            "eligible": not reasons,
            "exclusion_reasons": reasons,
            "eligibility_checks": checks,
        })
    table = pd.DataFrame(records).sort_values("station_id").reset_index(drop=True)
    return table[table["eligible"]].copy(), records


def _conventional_features(variant) -> dict:
    predictors = variant.dataframe[PREDICTORS].astype(float)
    missing_by_predictor = predictors.isna().mean()
    standard_deviations = predictors.std(ddof=0)
    skewness = predictors.skew().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    autocorrelations = []
    for column in PREDICTORS:
        series = predictors[column].dropna()
        autocorrelations.append(float(series.autocorr(lag=1)) if len(series) > 2 else 0.0)
    pairwise = predictors.corr().abs().to_numpy()
    pairwise_value = float(pairwise[np.triu_indices_from(pairwise, k=1)].mean())
    return {
        "observation_count": int(len(predictors)),
        "predictor_count": int(len(PREDICTORS)),
        "missing_cell_count": int(predictors.isna().sum().sum()),
        "missing_ratio": float(predictors.isna().mean().mean()),
        "missing_ratio_sd_by_predictor": float(missing_by_predictor.std(ddof=0)),
        "predictor_std_mean": float(standard_deviations.mean()),
        "predictor_skew_mean": float(skewness.mean()),
        "predictor_lag1_autocorrelation_mean": float(np.nan_to_num(autocorrelations, nan=0.0).mean()),
        "predictor_pairwise_correlation_abs": 0.0 if np.isnan(pairwise_value) else pairwise_value,
    }


def _station_cache_paths(station_id: str) -> tuple[Path, Path]:
    key = _stable_hash([CACHE_VERSION, station_id, str(RANDOM_SEED), repr(MISSINGNESS_LEVELS), str(VARIANTS_PER_LEVEL)])
    return CACHE_DIR / f"{key}_features.csv", CACHE_DIR / f"{key}_benchmark.csv"


def _build_station_cache(station_id: str) -> tuple[str, int, float]:
    feature_path, benchmark_path = _station_cache_paths(station_id)
    if feature_path.exists() and benchmark_path.exists():
        return station_id, int(pd.read_csv(benchmark_path, usecols=["variant_id"])["variant_id"].nunique()), 0.0

    started = time.perf_counter()
    station_df = load_french_piezo_station(DATASET_PATH, station_id)
    station_variants = generate_variants(
        station_df, station_id, MISSINGNESS_LEVELS, VARIANTS_PER_LEVEL, RANDOM_SEED
    )
    variant_features = []
    for variant in station_variants:
        try:
            profile = extract_dpp(variant, PREDICTORS)
            dpp = {column: float(getattr(profile, column)) for column in DPP_FEATURE_COLUMNS}
            dpp["dpp_extraction_status"] = "success"
            dpp["dpp_failure_reason"] = None
        except Exception as exc:
            dpp = {column: float("nan") for column in DPP_FEATURE_COLUMNS}
            dpp["dpp_extraction_status"] = "failed"
            dpp["dpp_failure_reason"] = str(exc)
        variant_features.append({
            "variant_id": variant.variant_id,
            "station_id": variant.station_id,
            "missingness_level": variant.missingness_level,
            "random_seed": variant.random_seed,
            **dpp,
            **_conventional_features(variant),
        })
    benchmark = evaluate_strategy_benchmark(station_variants, PREDICTORS, TARGET)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(variant_features).to_csv(feature_path, index=False)
    benchmark.to_csv(benchmark_path, index=False)
    return station_id, len(station_variants), time.perf_counter() - started


def _build_strengthened_dataset(max_stations: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(DATASET_PATH, usecols=["bss", "time", "tp", "e", "p"])
    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    eligible_table, screening_records = screen_stations(raw)
    station_ids = eligible_table["station_id"].tolist()
    if max_stations is not None:
        station_ids = station_ids[:max_stations]
        eligible_table = eligible_table[eligible_table["station_id"].isin(station_ids)].copy()
        screening_records = [record for record in screening_records if record["station_id"] in station_ids]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(os.environ.get("DPP_STRENGTHENED_WORKERS", "4")), len(station_ids)))
    stage_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        station_runs = list(executor.map(_build_station_cache, station_ids))
    benchmark_parts = []
    feature_parts = []
    generated_variant_count = 0
    for station_id, variant_count, _ in station_runs:
        feature_path, benchmark_path = _station_cache_paths(station_id)
        feature_parts.append(pd.read_csv(feature_path))
        benchmark_parts.append(pd.read_csv(benchmark_path))
        generated_variant_count += variant_count
    variant_features = pd.concat(feature_parts, ignore_index=True)
    benchmark = pd.concat(benchmark_parts, ignore_index=True)
    failure_rows = []
    summary_rows = benchmark.drop_duplicates(["variant_id", "strategy_name"]).copy()
    labels = []
    for variant_id, group in summary_rows.groupby("variant_id", sort=True):
        successful = group[group["status"] == "success"].dropna(subset=["mean_rmse"])
        strategies = set(successful["strategy_name"])
        if strategies != set(DEFAULT_STRATEGY_NAMES):
            failure_rows.append({
                "variant_id": variant_id,
                "station_id": str(group["station_id"].iloc[0]),
                "reason": "not_all_strategies_completed",
                "successful_strategies": sorted(strategies),
            })
            continue
        ranked = successful.sort_values(["mean_rmse", "strategy_name"]).reset_index(drop=True)
        labels.append({
            "variant_id": variant_id,
            "station_id": str(ranked["station_id"].iloc[0]),
            "missingness_level": float(ranked["missingness_level"].iloc[0]),
            "random_seed": int(ranked["random_seed"].iloc[0]),
            "best_strategy": str(ranked["strategy_name"].iloc[0]),
            "best_rmse": float(ranked["mean_rmse"].iloc[0]),
            "second_best_strategy": str(ranked["strategy_name"].iloc[1]),
            "second_best_rmse": float(ranked["mean_rmse"].iloc[1]),
            "best_second_rmse_margin": float(ranked["mean_rmse"].iloc[1] - ranked["mean_rmse"].iloc[0]),
        })
    label_frame = pd.DataFrame(labels).merge(pd.DataFrame(variant_features), on=["variant_id", "station_id", "missingness_level", "random_seed"], how="left")
    if label_frame.empty:
        raise RuntimeError("No complete labelled variants were produced.")
    return label_frame.sort_values("variant_id").reset_index(drop=True), benchmark, eligible_table, {
        "screening_records": screening_records,
        "failure_rows": failure_rows,
        "generated_variant_count": generated_variant_count,
        "benchmark_stage_seconds": time.perf_counter() - stage_started,
        "benchmark_workers": workers,
        "cache_directory": str(CACHE_DIR.relative_to(REPO_ROOT)),
    }


def _metric_dict(y_true, predictions, labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }


def _model_factories() -> dict[str, Callable[[], object]]:
    return {
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED),
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=RANDOM_SEED, n_jobs=1
        ),
    }


def _make_splits(labels: pd.DataFrame) -> list[dict]:
    y = labels["best_strategy"]
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    splits = []
    for index, (train, test) in enumerate(cv.split(labels, y)):
        train_ids = labels.iloc[train]["variant_id"].astype(str).tolist()
        test_ids = labels.iloc[test]["variant_id"].astype(str).tolist()
        splits.append({
            "fold_id": index,
            "repetition": index // N_SPLITS,
            "fold_within_repetition": index % N_SPLITS,
            "train_indices_hash": _stable_hash(train_ids),
            "test_indices_hash": _stable_hash(test_ids),
            "train_count": len(train_ids),
            "test_count": len(test_ids),
            "train": train,
            "test": test,
        })
    return splits


def _run_representation(labels: pd.DataFrame, feature_columns: list[str], factory, splits: list[dict]) -> dict:
    X = labels[feature_columns].fillna(0.0)
    y = labels["best_strategy"]
    label_order = sorted(y.unique().tolist())
    folds = []
    for split in splits:
        model = factory()
        model.fit(X.iloc[split["train"]], y.iloc[split["train"]])
        predictions = model.predict(X.iloc[split["test"]])
        metrics = _metric_dict(y.iloc[split["test"]], predictions, label_order)
        folds.append({"fold_id": split["fold_id"], **metrics})
    return {metric: {"mean": float(np.mean([row[metric] for row in folds])), "std": float(np.std([row[metric] for row in folds]))} for metric in ["accuracy", "macro_f1", "balanced_accuracy"]} | {"folds": folds}


def _run_majority(labels: pd.DataFrame, splits: list[dict]) -> dict:
    y = labels["best_strategy"]
    label_order = sorted(y.unique().tolist())
    folds = []
    for split in splits:
        majority = y.iloc[split["train"]].value_counts().idxmax()
        folds.append({"fold_id": split["fold_id"], **_metric_dict(y.iloc[split["test"]], [majority] * len(split["test"]), label_order)})
    return {metric: {"mean": float(np.mean([row[metric] for row in folds])), "std": float(np.std([row[metric] for row in folds]))} for metric in ["accuracy", "macro_f1", "balanced_accuracy"]} | {"folds": folds}


def _bootstrap_ci(differences: list[float], resamples: int = 1000) -> dict:
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    sampled = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return {"lower_95": float(np.percentile(sampled, 2.5)), "upper_95": float(np.percentile(sampled, 97.5))}


def _paired_comparison(left: dict, right: dict, metric: str) -> dict:
    left_values = [row[metric] for row in left["folds"]]
    right_values = [row[metric] for row in right["folds"]]
    differences = (np.asarray(left_values) - np.asarray(right_values)).tolist()
    return {
        "mean_difference": float(np.mean(differences)),
        "std_difference": float(np.std(differences)),
        "bootstrap_ci_95": _bootstrap_ci(differences),
        "wins": int(sum(value > 0 for value in differences)),
        "losses": int(sum(value < 0 for value in differences)),
        "ties": int(sum(value == 0 for value in differences)),
    }


def _fixed_and_oracle(labels: pd.DataFrame, benchmark: pd.DataFrame, splits: list[dict]) -> dict:
    y = labels["best_strategy"]
    label_order = sorted(y.unique().tolist())
    scores = benchmark.drop_duplicates(["variant_id", "strategy_name"])[["variant_id", "strategy_name", "mean_rmse"]]
    score_map = scores.pivot(index="variant_id", columns="strategy_name", values="mean_rmse")
    fixed_folds, oracle_folds = [], []
    for split in splits:
        train_ids = labels.iloc[split["train"]]["variant_id"]
        test_labels = labels.iloc[split["test"]]
        training_scores = score_map.loc[train_ids].mean(axis=0).sort_values()
        fixed_strategy = str(training_scores.index[0])
        fixed_folds.append({"fold_id": split["fold_id"], **_metric_dict(test_labels["best_strategy"], [fixed_strategy] * len(test_labels), label_order), "selected_strategy": fixed_strategy})
        oracle_folds.append({"fold_id": split["fold_id"], **_metric_dict(test_labels["best_strategy"], test_labels["best_strategy"], label_order)})
    def aggregate(folds):
        return {metric: {"mean": float(np.mean([row[metric] for row in folds])), "std": float(np.std([row[metric] for row in folds]))} for metric in ["accuracy", "macro_f1", "balanced_accuracy"]} | {"folds": folds}
    return {"fixed_preprocessing": aggregate(fixed_folds), "oracle": aggregate(oracle_folds)}


def _missingness_results(labels: pd.DataFrame, result: dict, splits: list[dict]) -> dict:
    values = {}
    for level in MISSINGNESS_LEVELS:
        level_folds = []
        for split, fold in zip(splits, result["folds"]):
            test = labels.iloc[split["test"]]
            mask = np.isclose(test["missingness_level"], level)
            if not mask.any():
                continue
            y_true = test.loc[mask, "best_strategy"]
            # Fold-level predictions are not retained in the compact model result;
            # rerun only the requested primary model for this predefined slice.
            X = labels[DPP_FEATURE_COLUMNS].fillna(0.0)
            model = _model_factories()["decision_tree"]()
            model.fit(X.iloc[split["train"]], labels["best_strategy"].iloc[split["train"]])
            predictions = model.predict(X.iloc[split["test"]][mask])
            level_folds.append(_metric_dict(y_true, predictions, sorted(labels["best_strategy"].unique())))
        values[str(level)] = {metric: {"mean": float(np.mean([row[metric] for row in level_folds])), "std": float(np.std([row[metric] for row in level_folds]))} for metric in ["accuracy", "macro_f1", "balanced_accuracy"]}
    return values


def _loso(labels: pd.DataFrame, feature_columns: list[str], factory) -> dict:
    X = labels[feature_columns].fillna(0.0)
    y = labels["best_strategy"]
    label_order = sorted(y.unique().tolist())
    results = {}
    for station_id in sorted(labels["station_id"].unique()):
        train = labels["station_id"] != station_id
        test = labels["station_id"] == station_id
        model = factory()
        model.fit(X.loc[train], y.loc[train])
        predictions = model.predict(X.loc[test])
        test_classes = sorted(y.loc[test].unique().tolist())
        train_classes = sorted(y.loc[train].unique().tolist())
        results[station_id] = {
            "variant_count": int(test.sum()),
            "training_station_count": int(labels.loc[train, "station_id"].nunique()),
            "class_distribution": {str(k): int(v) for k, v in y.loc[test].value_counts().to_dict().items()},
            "test_classes_absent_from_training": sorted(set(test_classes) - set(train_classes)),
            "metrics": _metric_dict(y.loc[test], predictions, label_order),
        }
    return results


def _label_stability(labels: pd.DataFrame) -> dict:
    margins = labels["best_second_rmse_margin"].to_numpy()
    return {
        "variant_count": int(len(labels)),
        "margin_mean": float(np.mean(margins)),
        "margin_median": float(np.median(margins)),
        "margin_sd": float(np.std(margins)),
        "threshold_counts": {str(threshold): int((margins <= threshold).sum()) for threshold in [0.001, 0.01, 0.05]},
        "threshold_fractions": {str(threshold): float((margins <= threshold).mean()) for threshold in [0.001, 0.01, 0.05]},
    }


def _ablation(labels: pd.DataFrame, splits: list[dict]) -> dict:
    full = _run_representation(labels, DPP_FEATURE_COLUMNS, _model_factories()["decision_tree"], splits)
    output = {"full_dpp": full}
    for descriptor in DPP_FEATURE_COLUMNS:
        columns = [column for column in DPP_FEATURE_COLUMNS if column != descriptor]
        result = _run_representation(labels, columns, _model_factories()["decision_tree"], splits)
        result["paired_delta_vs_full"] = {metric: _paired_comparison(result, full, metric) for metric in ["macro_f1", "balanced_accuracy"]}
        output[f"without_{descriptor}"] = result
    return output


def run_strengthened_experiment(max_stations: int | None = None) -> dict:
    study_started = time.perf_counter()
    labels, benchmark, eligible_table, build_info = _build_strengthened_dataset(max_stations=max_stations)
    benchmark_stage_seconds = build_info["benchmark_stage_seconds"]
    recommendation_started = time.perf_counter()
    splits = _make_splits(labels)
    models = _model_factories()
    pooled = {"majority": _run_majority(labels, splits)}
    for representation, columns in REPRESENTATIONS.items():
        pooled[representation] = {model_name: _run_representation(labels, columns, factory, splits) for model_name, factory in models.items()}
    fixed_oracle = _fixed_and_oracle(labels, benchmark, splits)
    comparisons = {}
    primary = pooled["full_dpp"]["decision_tree"]
    comparison_targets = {
        "majority": pooled["majority"],
        "missing_ratio_only": pooled["missing_ratio_only"]["decision_tree"],
        "conventional_meta_features": pooled["conventional_meta_features"]["decision_tree"],
        "fixed_preprocessing": fixed_oracle["fixed_preprocessing"],
    }
    for name, result in comparison_targets.items():
        comparisons[name] = {metric: _paired_comparison(primary, result, metric) for metric in ["accuracy", "macro_f1", "balanced_accuracy"]}
    variant_station_counts = labels.groupby("station_id").size().to_dict()
    benchmark_summary = []
    for strategy, group in benchmark.groupby("strategy_name"):
        unique = group.drop_duplicates(["variant_id", "strategy_name"])
        benchmark_summary.append({
            "strategy_name": strategy,
            "variant_count": int(len(unique)),
            "successful_fold_count": int((group["status"] == "success").sum()),
            "failed_fold_count": int((group["status"] == "failed").sum()),
            "mean_rmse": float(unique["mean_rmse"].mean()),
            "mean_mae": float(unique["mean_mae"].mean()),
            "mean_r2": float(unique["mean_r2"].mean()),
        })
    station_grouped = {
        "full_dpp_decision_tree": _loso(labels, DPP_FEATURE_COLUMNS, models["decision_tree"]),
        "missing_ratio_decision_tree": _loso(labels, ["missing_ratio"], models["decision_tree"]),
        "conventional_decision_tree": _loso(labels, CONVENTIONAL_FEATURE_COLUMNS, models["decision_tree"]),
        "full_dpp_logistic_regression": _loso(labels, DPP_FEATURE_COLUMNS, models["logistic_regression"]),
        "full_dpp_random_forest": _loso(labels, DPP_FEATURE_COLUMNS, models["random_forest"]),
    }
    station_results = {}
    for station_id, group in labels.groupby("station_id"):
        station_results[station_id] = {
            "variant_count": int(variant_station_counts[station_id]),
            "label_distribution": {str(k): int(v) for k, v in group["best_strategy"].value_counts().to_dict().items()},
            "loso_full_dpp_decision_tree": station_grouped["full_dpp_decision_tree"][station_id],
            "loso_fixed_baseline": None,
        }
    recommendation_stage_seconds = time.perf_counter() - recommendation_started
    return {
        "study": {
            "name": "strengthened_dpp_recommendation_study",
            "data_source": str(DATASET_PATH.relative_to(REPO_ROOT)),
            "predictors": PREDICTORS,
            "target": TARGET,
            "missingness_levels": MISSINGNESS_LEVELS,
            "variants_per_station": VARIANTS_PER_LEVEL * len(MISSINGNESS_LEVELS),
            "random_seed": RANDOM_SEED,
        },
        "eligibility_rule": {
            "minimum_observations": MIN_OBSERVATIONS,
            "minimum_span_days": MIN_SPAN_DAYS,
            "minimum_predictor_coverage": MIN_PREDICTOR_COVERAGE,
            "minimum_target_coverage": MIN_TARGET_COVERAGE,
            "chronological_fold_geometry": f"at least {BENCHMARK_FOLDS + 2} timestamped observations",
            "candidate_station_count": len(build_info["screening_records"]),
            "eligible_station_count": int(len(eligible_table)),
            "excluded_station_count": int(len(build_info["screening_records"]) - len(eligible_table)),
        },
        "dataset_summary": {
            "generated_variant_count": build_info["generated_variant_count"],
            "labelled_variant_count": int(len(labels)),
            "station_count": int(labels["station_id"].nunique()),
            "label_distribution": {str(k): int(v) for k, v in labels["best_strategy"].value_counts().to_dict().items()},
            "failed_variant_count": len(build_info["failure_rows"]),
        },
        "model_parameters": {
            "decision_tree": {"max_depth": 3, "random_state": RANDOM_SEED},
            "logistic_regression": {"max_iter": 1000, "random_state": RANDOM_SEED, "standardization": True},
            "random_forest": {"n_estimators": 200, "max_depth": 5, "random_state": RANDOM_SEED, "n_jobs": 1},
        },
        "validation": {"method": "RepeatedStratifiedKFold", "n_splits": N_SPLITS, "n_repeats": N_REPEATS, "random_state": RANDOM_SEED, "fold_count": len(splits)},
        "representations": REPRESENTATIONS,
        "pooled_results": pooled,
        "fixed_preprocessing_and_oracle": fixed_oracle,
        "paired_comparisons_full_dpp_decision_tree": comparisons,
        "station_grouped_results": station_grouped,
        "per_station_results": station_results,
        "missingness_level_results": _missingness_results(labels, primary, splits),
        "label_stability": _label_stability(labels),
        "descriptor_ablation": _ablation(labels, splits),
        "strategy_benchmark_summary": benchmark_summary,
        "screening_records": build_info["screening_records"],
        "failure_accounting": build_info["failure_rows"],
        "split_manifest": [{key: value for key, value in split.items() if key not in {"train", "test"}} for split in splits],
        "reproducibility": {"variant_ids_deterministic": True, "dpp_uses_target": False, "station_id_used_as_feature": False},
        "timing_seconds": {
            "benchmark_generation_and_cache_assembly": benchmark_stage_seconds,
            "recommendation_and_analysis": recommendation_stage_seconds,
            "total": time.perf_counter() - study_started,
            "benchmark_workers": build_info["benchmark_workers"],
            "cache_directory": build_info["cache_directory"],
        },
    }, labels, benchmark


def _write_outputs(summary: dict, labels: pd.DataFrame, benchmark: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUTPUT_DIR / "strengthened_dataset.csv", index=False)
    benchmark.to_csv(OUTPUT_DIR / "strengthened_strategy_benchmark.csv", index=False)
    pd.DataFrame(summary["screening_records"]).to_csv(OUTPUT_DIR / "strengthened_station_screening.csv", index=False)
    pd.DataFrame(summary["split_manifest"]).to_csv(OUTPUT_DIR / "strengthened_split_manifest.csv", index=False)
    fold_rows = []
    for representation, models in summary["pooled_results"].items():
        if representation == "majority":
            models = {"majority": models}
        for model_name, result in models.items():
            for fold in result["folds"]:
                fold_rows.append({"representation": representation, "model": model_name, **fold})
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "strengthened_pooled_fold_results.csv", index=False)
    (OUTPUT_DIR / "strengthened_experiment.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    rows = []
    for representation, models in summary["pooled_results"].items():
        if representation == "majority":
            models = {"majority": models}
        for model_name, result in models.items():
            rows.append({"representation": representation, "model": model_name, **{f"{metric}_mean": result[metric]["mean"] for metric in ["accuracy", "macro_f1", "balanced_accuracy"]}, **{f"{metric}_sd": result[metric]["std"] for metric in ["accuracy", "macro_f1", "balanced_accuracy"]}})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "strengthened_experiment.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stations", type=int, default=None)
    args = parser.parse_args()
    summary, labels, benchmark = run_strengthened_experiment(max_stations=args.max_stations)
    _write_outputs(summary, labels, benchmark)
    print(json.dumps(summary["dataset_summary"], indent=2, sort_keys=True))
    print(json.dumps(summary["eligibility_rule"], indent=2, sort_keys=True))
    print("Outputs written to outputs/experiments/strengthened_*")


if __name__ == "__main__":
    main()