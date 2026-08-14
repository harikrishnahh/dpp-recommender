from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from dpp_recommender.dataset.ingest import load_french_piezo_station
from dpp_recommender.descriptor_extraction import extract_dpp
from dpp_recommender.meta_dataset import build_meta_dataset, evaluate_majority_baseline, fit_recommendation_model
from dpp_recommender.strategy_evaluation import evaluate_strategy_benchmark
from dpp_recommender.variant_generator.generator import generate_variants


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPO_ROOT / "data" / "raw" / "dataset_2015_2021.csv"
STATION_ID = "00365X0003/P1"
PREDICTORS = ["tp", "e"]
TARGET_COLUMN = "p"
MISSINGNESS_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
VARIANTS_PER_LEVEL = 8
RANDOM_SEED = 42
MIN_CLASSES_REQUIRED = 2
MIN_MINORITY_VARIANTS = 5
PREFERRED_SECOND_STATIONS = [
    "00061X0117/PZ1",
    "00068X0147/PZ5",
]
MAX_SECOND_STATION_CANDIDATES = 10


def _best_strategy_distribution(benchmark: pd.DataFrame) -> dict:
    per_variant = benchmark.groupby(["variant_id", "strategy_name"], as_index=False).agg(
        mean_rmse=("mean_rmse", "first")
    )
    per_variant = per_variant.dropna(subset=["mean_rmse"])
    if per_variant.empty:
        return {}
    best_rows = per_variant.loc[per_variant.groupby("variant_id")["mean_rmse"].idxmin()].copy()
    counts = best_rows["strategy_name"].value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _distribution_is_meaningful(distribution: dict) -> bool:
    if len(distribution) < MIN_CLASSES_REQUIRED:
        return False
    return min(int(value) for value in distribution.values()) >= MIN_MINORITY_VARIANTS


def _candidate_stations(primary_station: str) -> list[str]:
    columns = ["bss", "time", "tp", "e", "p"]
    raw = pd.read_csv(DATASET_PATH, usecols=columns)
    raw = raw.dropna(subset=["bss", "time"]).copy()
    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    grouped = raw.groupby("bss").agg(
        n=("bss", "size"),
        tmin=("time", "min"),
        tmax=("time", "max"),
        tp_std=("tp", "std"),
        e_std=("e", "std"),
        p_std=("p", "std"),
    )
    grouped["span_days"] = (grouped["tmax"] - grouped["tmin"]).dt.days

    candidates = grouped[
        (grouped.index != primary_station)
        & (grouped["n"] >= 2000)
        & (grouped["span_days"] >= 2000)
        & (grouped["tp_std"] > 0)
        & (grouped["e_std"] > 0)
        & (grouped["p_std"] > 0)
    ].copy()
    if candidates.empty:
        raise ValueError("No suitable station candidates satisfy coverage and variation constraints.")

    # Deterministic station selection: strongest coverage first, then highest target variability.
    candidates = candidates.sort_values(["n", "span_days", "p_std"], ascending=[False, False, False])
    fallback = [str(station_id) for station_id in candidates.index.tolist()]
    ordered: list[str] = []
    for station in PREFERRED_SECOND_STATIONS:
        if station in fallback and station not in ordered:
            ordered.append(station)
    for station in fallback:
        if station not in ordered:
            ordered.append(station)
    return ordered[:MAX_SECOND_STATION_CANDIDATES]


def _run_for_stations(stations: List[str]) -> tuple[list, pd.DataFrame, dict]:
    variants = []
    for station in stations:
        station_df = load_french_piezo_station(DATASET_PATH, station)
        station_variants = generate_variants(
            dataframe=station_df,
            station_id=station,
            missingness_levels=MISSINGNESS_LEVELS,
            variants_per_level=VARIANTS_PER_LEVEL,
            random_seed=RANDOM_SEED,
        )
        variants.extend(station_variants)

    benchmark = evaluate_strategy_benchmark(variants, PREDICTORS, TARGET_COLUMN)
    distribution = _best_strategy_distribution(benchmark)
    return variants, benchmark, distribution


def _strategy_metrics(benchmark: pd.DataFrame) -> list[dict]:
    metrics = benchmark.groupby("strategy_name", as_index=False).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
    )
    return metrics.sort_values("rmse_mean").to_dict(orient="records")


def _dpp_variation(meta_dataset: pd.DataFrame, strategy_distribution: dict) -> dict:
    descriptor_columns = [
        "missing_ratio",
        "mean_gap_length",
        "mean_lag1_autocorrelation",
        "mean_trend_strength",
        "mean_absolute_pairwise_correlation",
    ]
    variant_dpp = meta_dataset[["variant_id", *descriptor_columns]].drop_duplicates(subset=["variant_id"]).copy()

    overall = {}
    for column in descriptor_columns:
        series = variant_dpp[column]
        overall[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "std": float(series.std(ddof=0)),
        }

    if len(strategy_distribution) < 2:
        return {"overall": overall, "by_best_strategy": {}}

    per_variant = meta_dataset.groupby(["variant_id", "strategy_name"], as_index=False).agg(
        mean_rmse=("mean_rmse", "first")
    )
    winners = per_variant.loc[per_variant.groupby("variant_id")["mean_rmse"].idxmin()][["variant_id", "strategy_name"]]
    merged = variant_dpp.merge(winners, on="variant_id", how="left")

    by_strategy = {}
    for strategy, group in merged.groupby("strategy_name"):
        by_strategy[str(strategy)] = {
            "count": int(len(group)),
            "descriptor_means": {
                column: float(group[column].mean())
                for column in descriptor_columns
            },
            "descriptor_stds": {
                column: float(group[column].std(ddof=0))
                for column in descriptor_columns
            },
        }

    return {"overall": overall, "by_best_strategy": by_strategy}


def main() -> None:
    output_dir = REPO_ROOT / "outputs" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    stations_used = [STATION_ID]
    variants, benchmark, strategy_distribution = _run_for_stations(stations_used)

    # Step 1 stop condition: keep one station only if class diversity is meaningful.
    if not _distribution_is_meaningful(strategy_distribution):
        for candidate in _candidate_stations(STATION_ID):
            if candidate in stations_used:
                continue
            tentative_stations = stations_used + [candidate]
            tentative_variants, tentative_benchmark, tentative_distribution = _run_for_stations(tentative_stations)
            if _distribution_is_meaningful(tentative_distribution):
                stations_used = tentative_stations
                variants = tentative_variants
                benchmark = tentative_benchmark
                strategy_distribution = tentative_distribution
                break

    profile_map = {variant.variant_id: extract_dpp(variant, PREDICTORS) for variant in variants}
    meta_dataset = build_meta_dataset(variants, benchmark, profile_map, dataset_name="FrenchPiezo")
    recommendation = fit_recommendation_model(meta_dataset, random_state=42)
    majority_baseline = evaluate_majority_baseline(
        meta_dataset,
        random_state=RANDOM_SEED,
        train_variants=recommendation.get("train_variants"),
        test_variants=recommendation.get("test_variants"),
    )

    benchmark_path = output_dir / "strategy_benchmark_expanded.csv"
    meta_path = output_dir / "meta_dataset_expanded.csv"
    recommendation_path = output_dir / "recommendation_summary_expanded.json"

    benchmark.to_csv(benchmark_path, index=False)
    meta_dataset.to_csv(meta_path, index=False)

    unique_best_strategies = sorted(strategy_distribution.keys())
    recommendation_multiclass = len(unique_best_strategies) >= 2
    dpp_stats = _dpp_variation(meta_dataset, strategy_distribution)
    strategy_metrics = _strategy_metrics(benchmark)

    summary = {
        "dataset_name": "FrenchPiezo",
        "stations_used": stations_used,
        "predictors": PREDICTORS,
        "target": TARGET_COLUMN,
        "missingness_levels": MISSINGNESS_LEVELS,
        "variants_per_level": VARIANTS_PER_LEVEL,
        "random_seed": RANDOM_SEED,
        "n_variants": len(variants),
        "n_strategy_rows": len(benchmark),
        "n_meta_rows": len(meta_dataset),
        "best_strategy": recommendation.get("best_strategy"),
        "majority_baseline": {
            "accuracy": majority_baseline.get("accuracy"),
            "balanced_accuracy": majority_baseline.get("balanced_accuracy"),
            "macro_f1": majority_baseline.get("macro_f1"),
            "label_order": majority_baseline.get("label_order"),
            "confusion_matrix": majority_baseline.get("confusion_matrix"),
            "per_class_metrics": majority_baseline.get("per_class_metrics"),
            "number_of_training_variants": majority_baseline.get("number_of_training_variants"),
            "number_of_test_variants": majority_baseline.get("number_of_test_variants"),
        },
        "dpp_recommendation": {
            "accuracy": recommendation.get("recommendation_accuracy"),
            "balanced_accuracy": recommendation.get("recommendation_balanced_accuracy"),
            "macro_f1": recommendation.get("recommendation_macro_f1"),
            "number_of_training_variants": recommendation.get("number_of_training_variants"),
            "number_of_test_variants": recommendation.get("number_of_test_variants"),
            "confusion_matrix": recommendation.get("confusion_matrix"),
            "label_order": recommendation.get("label_order"),
            "per_class_metrics": recommendation.get("per_class_metrics"),
            "train_variants": recommendation.get("train_variants", []),
            "test_variants": recommendation.get("test_variants", []),
        },
        "forecasting_strategy_metrics": strategy_metrics,
        "dpp_variation_analysis": dpp_stats,
        "strategy_rankings": recommendation.get("strategy_rankings", {}),
        "strategy_class_distribution": recommendation.get("strategy_class_distribution", {}),
        "empirical_best_strategy_distribution": strategy_distribution,
        "unique_best_strategy_classes": unique_best_strategies,
        "recommendation_target_is_multiclass": recommendation_multiclass,
        "meets_meaningful_diversity_threshold": _distribution_is_meaningful(strategy_distribution),
    }
    recommendation_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
